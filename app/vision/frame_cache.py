"""Private, bounded half-second snapshots shared with the anatomy decode.

Only exact half-second grid requests at <=1024px use this cache.
Stored 1024px snapshots supply both 768px discovery and 1024px rechecks of
EXACTLY the same decoded frame; never upscale a discovery thumbnail. Shot/recheck
requests off that grid still use exact seeking. FPS sampling can differ by up
to a source-frame/half sampling interval from a seek; evidence records this
sampling protocol and checkpoints are fingerprinted separately.
"""
import asyncio
import base64
import hashlib
import io
import math
from pathlib import Path
import tempfile

VERSION = 'fps-cache-v1'


class FrameCache:
    def __init__(self, root, fingerprint, video_path, ffmpeg_bin, duration, *, local_gate_owned=False, max_bytes=256*1024*1024):
        try:
            from PIL import Image  # Fail before contacting any worker if the app image is stale.
        except ImportError:
            raise ValueError('Contextual frame cache requires Pillow; rebuild the application image from updated requirements') from None
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.identity = hashlib.sha256((VERSION+fingerprint).encode()).hexdigest()
        self.video_path,self.ffmpeg_bin,self.duration = video_path,ffmpeg_bin,duration
        self.local_gate_owned,self.max_bytes = local_gate_owned,max_bytes
        self.lock = asyncio.Lock()
        self.hits = self.fallback_chunks = 0
        self.decoded_chunks=set()
        self.entries={p:(p.stat().st_size,p.stat().st_mtime) for p in self.root.glob('*.jpg') if not p.is_symlink()}
        self.size=sum(s for s,_ in self.entries.values())

    def key(self, t):
        if not math.isfinite(t) or t<0 or t>self.duration or abs(t-(round(t-.5)+.5))>1e-5:
            return None
        return self.root/(self.identity+'-'+str(round(t-.5))+'.jpg')

    def get(self,t):
        path=self.key(t)
        if path is None or path.is_symlink() or not path.is_file(): return None
        digest=path.with_suffix('.sha256')
        if digest.is_symlink() or not digest.is_file(): return None
        if path.stat().st_size>1024*1024 or digest.stat().st_size!=64:return None
        data=path.read_bytes()
        if len(data)>1024*1024 or hashlib.sha256(data).hexdigest()!=digest.read_text(): return None
        path.touch()
        if path in self.entries:self.entries[path]=(len(data),path.stat().st_mtime)
        self.hits+=1
        return data

    def add(self,t,data):
        path=self.key(t)
        if path is None: return
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            if image.width>8192 or image.height>8192: raise ValueError('Invalid cache image dimensions')
            image.thumbnail((1024,1024))
            output=io.BytesIO()
            image.convert('RGB').save(output,format='JPEG',quality=85)
        data=output.getvalue()
        if len(data)>1024*1024 or len(data)>self.max_bytes: raise ValueError('Cache image budget exceeded')
        previous=self.entries.pop(path,(0,0))[0]
        self.size-=previous
        while self.entries and self.size+len(data)>self.max_bytes:
            old=min(self.entries,key=lambda p:self.entries[p][1])
            self.size-=self.entries.pop(old)[0]
            old.unlink(missing_ok=True);old.with_suffix('.sha256').unlink(missing_ok=True)
        # No partially written JPEG can become a successful cache hit.
        part=path.with_suffix('.part')
        part.write_bytes(data);part.replace(path)
        digest=path.with_suffix('.digest-part')
        digest.write_text(hashlib.sha256(data).hexdigest());digest.replace(path.with_suffix('.sha256'))
        self.entries[path]=(len(data),path.stat().st_mtime)
        self.size+=len(data)

    def add_file(self,t,path):
        if self.key(t) is not None: self.add(t,Path(path).read_bytes())

    def accept_remote(self,data,scores):
        if data.get('sample_protocol')!=VERSION: raise ValueError('Invalid frame cache protocol')
        samples=data.get('sample_frames')
        if not isinstance(samples,list) or len(samples)>64: raise ValueError('Invalid frame cache samples')
        allowed={round(float(s['t']),6) for s in scores if self.key(float(s['t'])) is not None}
        parsed=[]
        for item in samples:
            t=float(item['t'])
            if round(t,6) not in allowed or not isinstance(item.get('jpeg'),str) or len(item['jpeg'])>1400000:
                raise ValueError('Invalid frame cache sample')
            allowed.remove(round(t,6))
            parsed.append((t,base64.b64decode(item['jpeg'],validate=True)))
        if allowed: raise ValueError('Incomplete shared frame cache')
        # Validate every image before exposing any of this response's samples.
        from PIL import Image
        for _,image in parsed:
            with Image.open(io.BytesIO(image)) as opened:
                if opened.format!='JPEG' or max(opened.size)>1024: raise ValueError('Invalid remote JPEG sample')
                opened.load()
        for t,image in parsed:self.add(t,image)

    async def frame(self,t,max_edge=768):
        if max_edge not in (768,1024) or self.key(t) is None:return None
        def resized(data):
            if data is None or max_edge==1024:return data
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                image.thumbnail((max_edge,max_edge))
                output=io.BytesIO();image.convert('RGB').save(output,format='JPEG',quality=85)
                return output.getvalue()
        async with self.lock:
            data=self.get(t)
            if data is not None:return resized(data)
            # Legacy workers/checkpoint resumes may have no shared snapshots.
            # Decode a bounded chunk once, never four seeks per base window.
            from app.vision.classifier import _extract_frames_continuous
            from app.queue.heavy_pipeline_lock import HEAVY_PIPELINE_LOCK
            from contextlib import nullcontext
            start=math.floor(t/60)*60
            duration=min(60,self.duration-start)
            if start in self.decoded_chunks:return None
            with tempfile.TemporaryDirectory() as tmp:
                async with (nullcontext() if self.local_gate_owned else HEAVY_PIPELINE_LOCK):
                    await asyncio.wait_for(_extract_frames_continuous(self.ffmpeg_bin,self.video_path,.25,Path(tmp),
                        duration_seconds=duration,start_offset=start,clip_duration=duration),timeout=120)
                for index,path in enumerate(sorted(Path(tmp).glob('frame_*.jpg'))):
                    self.add_file(start+index*.25,path)
            self.fallback_chunks+=1
            self.decoded_chunks.add(start)
            # At EOF an FPS filter may emit fewer frames than seeking can
            # retrieve. This is a cache miss, never a negative observation.
            return resized(self.get(t))
