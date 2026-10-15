"""HTTP/static deployment and transport boundary tests, without a DDS domain."""
from queue import Queue

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from web.server import create_app


def setup():
    requests,responses,states=Queue(64),Queue(64),Queue(1)
    app=create_app({'state_timeout':.5,'model_url':'/robot/urdf/little_white_v3.urdf'},requests,responses,states)
    return app,requests,responses,states


def test_offline_resources_are_served_from_same_origin():
    app,*_=setup()
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/config').json()['state_timeout']==.5
        assert '<robot name="little_white_v3">' in client.get('/robot/urdf/little_white_v3.urdf').text
        assert len(client.get('/robot/meshes/part_001_solid_001.stl').content)>1000
        assert 'Apache License' in client.get('/licenses/urdf-loader.txt').text
        assert client.get('/robot/../../little_white_wtw.yaml').status_code==404


def test_cross_origin_browser_socket_is_rejected():
    app,*_=setup()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws',headers={'origin':'http://unrelated.example'}):pass


@pytest.mark.parametrize('raw',['[]','{"type":"control","vx":NaN}','{"id":[]}', 'x'])
def test_invalid_frames_never_enter_controller_queue(raw):
    app,requests,*_=setup()
    with TestClient(app) as client:
        with client.websocket_connect('/ws',headers={'origin':'http://testserver'}) as ws:
            assert ws.receive_json()['type']=='hello'
            ws.send_text(raw)
            assert ws.receive_json()['ok'] is False
            assert requests.empty()


def test_queue_backpressure_is_explicit_and_disconnect_releases_session():
    app,requests,responses,_=setup()
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            hello=ws.receive_json();owner=hello['client']
            ws.send_json({'type':'acquire','id':1})
            packet=requests.get(timeout=1)
            assert packet[0]==owner and packet[1]['type']=='acquire'
            responses.put((owner,{'type':'result','id':1,'ok':True}))
            assert ws.receive_json()['id']==1
            for _ in range(64):requests.put(('full',{},0))
            ws.send_json({'type':'mode','mode':'stand','id':2})
            result=ws.receive_json()
            assert not result['ok'] and 'busy' in result['error']
            while not requests.empty():requests.get_nowait()
        assert requests.get(timeout=1)[1]['type']=='release'
