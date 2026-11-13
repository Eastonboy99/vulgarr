import asyncio
from contextlib import asynccontextmanager
import socket

from fastapi import FastAPI
import httpx
import pytest

from app.db.session import DEFAULT_SETTINGS
from app.routers import settings, setup
from app.scenes import protection
from app.vision import contextual


@pytest.mark.parametrize('invalid',[None,'public','semantic','folders'])
def test_opt_in_validates_local_endpoint_and_clean_output_without_starting_jobs(tmp_path,monkeypatch,invalid):
    stored={}
    @asynccontextmanager
    async def session():yield None
    async def get(session,key):return stored.get(key,DEFAULT_SETTINGS[key])
    async def set(session,key,value):stored[key]=value
    for module in (settings,setup):
        monkeypatch.setattr(module,'get_session',session)
        monkeypatch.setattr(module,'get_setting',get)
        monkeypatch.setattr(module,'set_setting',set)
    monkeypatch.setattr(protection,'get_setting',get)
    monkeypatch.setattr(contextual.socket,'getaddrinfo',lambda host,*args:[(socket.AF_INET,1,6,'',('8.8.8.8' if host=='public' else '127.0.0.1',1234))])
    app=FastAPI();app.include_router(settings.router);app.include_router(setup.router)
    form={'concurrency_cap':'1','off_hours_start':'01:00','off_hours_end':'07:00',
        'scene_protection_present':'true','scene_protection_enabled':'true',
        'movie_output_folder':str(tmp_path/'clean'), 'scene_contextual_present':'true',
        'scene_contextual_base_url':'http://localhost:1234/v1','scene_contextual_model':'mock'}
    if invalid=='public':form['scene_contextual_base_url']='http://public/v1'
    if invalid=='semantic':form['scene_semantic_enabled']='true';form['claude_vision_base_url']='http://localhost/v1'
    if invalid=='folders':form['movie_output_folder']=''
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://test') as client:
            response=await client.post('/settings',data=form)
            if invalid:
                assert response.status_code==422 and not stored
                return
            assert response.status_code==200 and stored['scene_protection_enabled'] is True
            config=await protection.context_config(None)
            assert config['scene_contextual_enabled'] is True and config['scene_contextual_window_seconds']==4
            assert 'Protection-first' in response.text
            # Omitted new checkbox fields must not silently disable automation.
            old=dict(form);old.pop('scene_protection_present');old.pop('scene_protection_enabled')
            assert (await client.post('/settings',data=old)).status_code==200
            assert stored['scene_protection_enabled'] is True
            before=dict(stored)
            for bad in ({'scene_protection_enabled':'true'}, {'scene_semantic_enabled':True},
                        {'scene_contextual_base_url':'http://public/v1'}, {'movie_output_folder':''}):
                response=await client.post('/setup/import',json=bad)
                assert response.status_code==400 and stored==before
            exported=(await client.get('/settings/export')).json()
            assert exported['scene_protection_enabled'] is True
    asyncio.run(scenario())
