"""Compare real DDS controller timing without/with ten local web viewers.

python tests/benchmark_web.py [--domain-id 72] [--seconds 5]
Reports measurements, not hard realtime guarantees on a shared host.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def measure(c,publish,infer,seconds,viewers=0,port=18889):
    from websockets.asyncio.client import connect
    sockets=[];readers=[]
    async def read(ws):
        async for _ in ws:pass
    for _ in range(viewers):
        ws=await connect(f'ws://127.0.0.1:{port}/ws',proxy=None)
        sockets.append(ws);readers.append(asyncio.create_task(read(ws)))
    await asyncio.sleep(1)
    publish.clear();infer.clear();start=time.monotonic()
    await asyncio.sleep(seconds)
    elapsed=time.monotonic()-start
    import numpy as np
    def stats(values):
        delta=np.diff(values)*1000
        return dict(hz=round(len(values)/elapsed,2),
                    p50_ms=round(float(np.percentile(delta,50)),3),
                    p95_ms=round(float(np.percentile(delta,95)),3),
                    p99_ms=round(float(np.percentile(delta,99)),3))
    result=dict(publish=stats(publish),policy=stats(infer),viewers=viewers)
    for ws in sockets:await ws.close()
    for reader in readers:reader.cancel()
    await asyncio.gather(*readers,return_exceptions=True)
    assert c.running,c.error
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain-id',type=int,default=72)
    parser.add_argument('--seconds',type=float,default=5);args=parser.parse_args()
    from little_white_wtw import WTWPolicy,Controller
    from web.bridge import WebBridge
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    output=ROOT/'web/test-results';output.mkdir(exist_ok=True)
    c=None;bridge=None
    with (output/'benchmark-simulator.log').open('w') as log:
        sim=subprocess.Popen([sys.executable,str(ROOT/'tests/run_web_smoke.py'),'--simulator',
                              '--domain-id',str(args.domain_id)],stdout=log,stderr=subprocess.STDOUT,
                              env={**os.environ,'SDL_VIDEODRIVER':'dummy'})
        try:
            runtime=WTWPolicy();ChannelFactoryInitialize(args.domain_id,'lo')
            c=Controller(runtime,use_gui=False);c.Init()
            publish=[];infer=[]
            original_publish=c._publish;original_infer=runtime.infer
            def record_publish(target=None):
                publish.append(time.monotonic());return original_publish(target)
            def record_infer(q,dq,quat):
                infer.append(time.monotonic());return original_infer(q,dq,quat)
            c._publish=record_publish;runtime.infer=record_infer
            c.set_mode('move')
            baseline=asyncio.run(measure(c,publish,infer,args.seconds))
            bridge=WebBridge(c,'127.0.0.1',18889)
            dashboard=asyncio.run(measure(c,publish,infer,args.seconds,viewers=10))
            result=dict(baseline=baseline,dashboard=dashboard)
            (output/'timing.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
        finally:
            if bridge:bridge.close()
            if c:c.ShutDown()
            sim.send_signal(signal.SIGINT)
            try:sim.wait(timeout=3)
            except subprocess.TimeoutExpired:sim.kill();sim.wait()

if __name__=='__main__':main()
