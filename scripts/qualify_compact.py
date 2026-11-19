"""Bounded, private image-only comparison; never scans/renders/publishes a title."""
import argparse
import asyncio
import json
import hashlib
import os
from pathlib import Path
import re
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.vision import compact,contextual


def load_cases(path,performance_only=False):
    path=Path(path)
    value=json.loads(path.read_text(encoding='utf-8'))
    cases=value['cases']
    if not isinstance(cases,list) or not 1<=len(cases)<=32:raise ValueError('Require 1-32 cases')
    seen=set()
    for case in cases:
        name=case['id']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name) or name in seen:
            raise ValueError('Invalid/duplicate case ID')
        seen.add(name)
        if not isinstance(case['frames'],list) or len(case['frames'])!=4:raise ValueError('Four frames required')
        frames=[]
        for frame in case['frames']:
            p=path.parent/Path(frame)
            if not p.is_file() or not 0<p.stat().st_size<=4*1024*1024:raise ValueError('Invalid bounded image file')
            frames.append(p)
        case['_paths']=frames
        if not performance_only:
            if (case.get('annotation')!='human' or not isinstance(case.get('states'),list)
                    or len(case['states'])!=4 or any(s not in contextual.STATES for s in case['states'])):
                raise ValueError('Accuracy comparisons require explicit human per-frame annotations')
    return cases


def metrics(reference,predicted):
    positive=contextual.POSITIVE
    blocking=positive|contextual.AMBIGUOUS
    return {'missed_positive_frames':sum(r in positive and p not in blocking for r,p in zip(reference,predicted)),
            'extra_flagged_frames':sum(r not in blocking and p in blocking for r,p in zip(reference,predicted)),
            'state_disagreements':sum(r!=p for r,p in zip(reference,predicted)),
            'ambiguous_reference_frames':sum(r in contextual.AMBIGUOUS for r in reference)}


async def run(args):
    cases=load_cases(args.manifest,args.performance_only)
    if not 0<args.wall_seconds<=600:raise ValueError('Wall budget must be 1-600 seconds')
    if args.output.suffix.lower()!='.json' or args.output.exists():raise ValueError('Require a new .json output file')
    results=[];started=time.monotonic()
    for index,case in enumerate(cases):
        images=[p.read_bytes() for p in case['_paths']]
        # Balance first/second request effects instead of always giving the
        # compact variant a potentially warm vision cache.
        for position,detailed in enumerate((True,False) if index%2==0 else (False,True)):
            remaining=args.wall_seconds-(time.monotonic()-started)
            if remaining<=0:raise ValueError('Qualification wall budget exhausted; no success report written')
            result=await compact.classify_frames(images,base_url=args.base_url,model=args.model,
                api_key=os.environ.get(args.api_key_env,'') if args.api_key_env else '',max_edge=args.edge,
                timeout=min(60,remaining),detailed=detailed)
            result['case_id']=case['id']
            result['position_in_pair']=position+1
            result['input_sha256']=[hashlib.sha256(image).hexdigest() for image in images]
            if not args.performance_only:result['reference_comparison']=metrics(case['states'],result['states'])
            results.append(result)
    report={'schema_version':1,'experimental':True,'accuracy_qualified':False,
            'mode':'performance_only' if args.performance_only else 'declared_human_reference_comparison',
            'publication_allowed':False,'request_count':len(results),'results':results,
            'total_wall_seconds':time.monotonic()-started,'image_edge':args.edge,
            'median_request_seconds':{variant:statistics.median(r['request_seconds'] for r in results if r['variant']==variant)
                                      for variant in ('detailed','compact')},
            'cache_caveat':'Paired requests may reuse vision/KV caches. These are warm comparisons, not cold streaming throughput.',
            'caveat':'Tiny corpus results do not qualify library recall, temporal boundaries, HDR or rendered-output assurance.'}
    with args.output.open('x',encoding='utf-8') as output:json.dump(report,output,indent=2)
    print(json.dumps({'requests':len(results),'median_request_seconds':report['median_request_seconds'],
                      'accuracy_qualified':False,'publication_allowed':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:1234')
    parser.add_argument('--model',required=True,help='Existing local model identifier; no weights are downloaded by this script')
    parser.add_argument('--api-key-env',default='')
    parser.add_argument('--edge',type=int,choices=(384,768),default=768)
    parser.add_argument('--wall-seconds',type=float,default=300)
    parser.add_argument('--performance-only',action='store_true')
    args=parser.parse_args()
    try:asyncio.run(run(args))
    except Exception:
        parser.exit(1,'Qualification failed: check local endpoint, manifest, strict model output and budgets. No success report/publication produced.\n')


if __name__=='__main__':main()
