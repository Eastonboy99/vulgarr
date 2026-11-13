"""Opt-in local qualification on independently annotated ORIGINAL clips.

Run in the application environment with media mounted and LM Studio configured:
  python scripts/qualify_protection.py --manifest cases.json --report report.json
  ... --render --work-dir /large/private/non-library/workspace

Manifest: {"cases": [{"name": "clothed control", "path": "/media/control.mkv",
                      "blocked": []}]}

Uses the actual scan pipeline in an isolated temporary DB. Does not modify the
application DB, enqueue jobs, download models, restart services or publish to a
viewing library. --render ALSO checks/corrects a private temporary output; this is
VISUAL qualification, not a profanity benchmark. Audio outside visual mute windows
is the original. Select a workspace OUTSIDE Plex with enough room before rendering.
A report is meaningful only with independent, accurate interval annotations.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.db.models import Base, Title, SceneJob, SceneJobKind, JobState, MediaType, DetectedScene
from app.db.session import DEFAULT_SETTINGS, get_session, get_setting, set_setting
from app.scenes import pipeline
from app.scenes.content_policy import scene_decision, Action
from app.scenes.qualification import interval_metrics
from app.mux.remux import probe


async def run(args):
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    cases=manifest['cases']
    if not isinstance(cases,list) or not cases:
        raise ValueError('At least one annotated case is required')
    async with get_session() as session:
        preferences={key:await get_setting(session,key) for key in DEFAULT_SETTINGS if key.startswith('scene_') or key.startswith('blur_video_')}
        clean_roots=[await get_setting(session,key) for key in ('movie_output_folder','tv_output_folder')]
    sources=[Path(case['path']).resolve() for case in cases]
    if (args.report.suffix.lower()!='.json' or args.report.resolve()==args.manifest.resolve()
            or any(args.report.resolve()==s or (args.report.exists() and args.report.samefile(s)) for s in sources)):
        raise ValueError('Report must be a separate JSON file, never a source or the annotation manifest')
    if args.render:
        workspace=args.work_dir.resolve()
        roots=[Path(r).resolve() for r in clean_roots if r]+[pipeline.app_settings.media_root.resolve()]
        if any(workspace==root or root in workspace.parents for root in roots):
            raise ValueError('Rendering qualification requires a workspace outside configured source and clean libraries')
        workspace.mkdir(parents=True,exist_ok=True)
    preferences.update(scene_protection_enabled=True,scene_semantic_enabled=False)
    reports=[]
    for case in cases:
        source=Path(case['path'])
        if not source.is_absolute() or not source.is_file():
            raise ValueError('Every qualification source must be an existing absolute local path')
        interval_metrics(case['blocked'],[])  # Validate annotations BEFORE inference.
        start=time.monotonic()
        with tempfile.TemporaryDirectory(prefix='.vulgarr-qualification-',dir=args.work_dir) as folder:
            work=Path(folder)
            engine=create_async_engine(f'sqlite+aiosqlite:///{work/"scratch.db"}')
            previous=pipeline.app_settings
            pipeline.app_settings=SimpleNamespace(**{**vars(previous),'data_dir':work})
            try:
                async with engine.begin() as connection:await connection.run_sync(Base.metadata.create_all)
                factory=async_sessionmaker(engine,expire_on_commit=False)
                async with factory() as session:
                    for key,value in preferences.items():await set_setting(session,key,value)
                    title=Title(id=1,media_type=MediaType.movie,display_name=case['name'],video_path=str(source),scene_scan_generation=1)
                    session.add(title);session.add(SceneJob(id=1,title_id=1,kind=SceneJobKind.scan,state=JobState.processing))
                    await session.commit()
                    duration=float((await probe(previous.ffprobe_bin,source))['format']['duration'])
                    if any(b>duration for a,b in case['blocked']):raise ValueError('Annotations extend beyond clip duration')
                    await pipeline.scan_for_scenes(session,title,1)
                    scan_seconds=time.monotonic()-start
                    if args.render:
                        await pipeline.apply_scene_blur(session,title,'qualification',output_path=work/'checked.mkv',audio_source_path=source)
                    rows=(await session.execute(select(DetectedScene))).scalars().all()
                    policy=preferences['scene_content_policy']
                    predicted=[(max(0,r.start_seconds-float(preferences['scene_blur_pad_start_seconds'])),
                                min(duration,r.end_seconds+float(preferences['scene_blur_pad_end_seconds'])))
                               for r in rows if scene_decision(r,policy).action!=Action.allow]
                    reports.append({'name':case['name'],'duration_seconds':duration,'scan_seconds':round(scan_seconds,3),
                        'total_seconds':round(time.monotonic()-start,3),'render_checked':args.render,
                        **interval_metrics(case['blocked'],predicted)})
            finally:
                pipeline.app_settings=previous
                await engine.dispose()
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({'qualification':'sampled, independently annotated visual benchmark','complete':len(reports)==len(cases),'requested_cases':len(cases),'completed_cases':len(reports),'cases':reports},indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(reports),'report':str(args.report),'render_checked':args.render}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--work-dir',type=Path)
    args=parser.parse_args()
    if args.render and not args.work_dir:parser.error('--render requires --work-dir outside the viewing library')
    asyncio.run(run(args))
