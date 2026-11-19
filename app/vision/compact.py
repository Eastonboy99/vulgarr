"""Experimental local state-only classifier. NOT publication evidence.

There is deliberately no pipeline/settings switch: qualification must precede
adoption. Output cannot pass content_policy.validate_evidence().
"""
import base64
import io
import json
import time

import httpx
from app.vision import contextual

VERSION = 'compact-states-experimental-v1'
PROMPT = (contextual.PROMPT.split('Return one observation')[0]
          .replace('Give brief non-graphic factual evidence.', 'Return only categorical states, no rationale.')
          .replace('Explain the actual evidence.', '')
          .replace(' and note others in evidence', '')
          + 'Return JSON with exactly four frames, each with only frame_id and state. '
            'No summary, probabilities, reasoning, approval or filtering instructions.')


class CompactAnalysisError(ValueError):
    pass


def schema():
    return {'type':'object','properties':{'frames':{'type':'array','minItems':4,'maxItems':4,
        'items':{'type':'object','properties':{'frame_id':{'type':'string','enum':['f0','f1','f2','f3']},
        'state':{'type':'string','enum':list(contextual.STATES)}},
        'required':['frame_id','state'],'additionalProperties':False}}},
        'required':['frames'],'additionalProperties':False}


def parse(raw):
    if not isinstance(raw,dict) or set(raw)!={'frames'}:raise CompactAnalysisError('Invalid compact fields')
    rows=raw['frames']
    if not isinstance(rows,list) or len(rows)!=4 or any(not isinstance(r,dict) for r in rows):
        raise CompactAnalysisError('Invalid compact frames')
    if any(set(r)!={'frame_id','state'} or not isinstance(r.get('frame_id'),str)
           or r['state'] not in contextual.STATES for r in rows):
        raise CompactAnalysisError('Invalid compact states')
    if sorted(r['frame_id'] for r in rows)!=['f0','f1','f2','f3']:
        raise CompactAnalysisError('Invalid compact frame identity')
    return sorted(rows,key=lambda r:r['frame_id'])


def prepare_image(data,max_edge):
    from PIL import Image
    if max_edge not in (384,768) or len(data)>4*1024*1024:raise ValueError('Invalid bounded image')
    with Image.open(io.BytesIO(data)) as image:
        if image.width*image.height>16_000_000:raise ValueError('Image dimensions exceed qualification budget')
        image.thumbnail((max_edge,max_edge))
        output=io.BytesIO();image.convert('RGB').save(output,format='JPEG',quality=85)
        return output.getvalue()


async def classify_frames(images,*,base_url,model,api_key='',max_edge=768,timeout=60,detailed=False):
    """Four existing images only; no discovery, approval, model download or DB."""
    try:
        if len(images)!=4 or not isinstance(model,str) or not model.strip() or not 0<timeout<=120:
            raise ValueError('Invalid qualification request')
        target,headers,extensions=await contextual.local_endpoint(base_url)
        if api_key:headers['Authorization']='Bearer '+api_key
        content=[{'type':'text','text':contextual.PROMPT if detailed else PROMPT}]
        for i,data in enumerate(images):
            image=prepare_image(data,max_edge)
            content.extend([{'type':'text','text':f'frame_id=f{i}'},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(image).decode()}}])
        started=time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout,connect=min(10,timeout)),trust_env=False,follow_redirects=False) as client:
            response=await client.post(target,headers=headers,extensions=extensions,json={
                'model':model,'messages':[{'role':'user','content':content}],
                'temperature':0,'max_tokens':2048 if detailed else 256,'reasoning_effort':'none',
                'response_format':{'type':'json_schema','json_schema':{'name':'detailed_qualification' if detailed else 'compact_qualification','strict':True,
                    'schema':contextual.response_schema() if detailed else schema()}}})
        if response.status_code!=200:raise CompactAnalysisError(f'Local qualification endpoint HTTP {response.status_code}')
        choice=response.json()['choices'][0]
        if choice.get('finish_reason')!='stop':raise CompactAnalysisError('Incomplete qualification response rejected')
        raw=json.loads(choice['message']['content'])
        if detailed:
            contextual.observations_to_evidence(raw,model)
            rows=sorted(raw['frames'],key=lambda r:r['frame_id'])
        else:rows=parse(raw)
        return {'protocol':VERSION,'experimental':True,'variant':'detailed' if detailed else 'compact',
                'model':model,'states':[r['state'] for r in rows],'request_seconds':time.monotonic()-started}
    except CompactAnalysisError:raise
    except Exception:
        raise CompactAnalysisError('Local qualification failed; invalid evidence/endpoint or request timeout') from None
