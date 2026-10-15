"""Optional FastAPI server, running in a spawned process without DDS access."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
from queue import Empty
import shlex
import sys
import time
import uuid

from .bridge import offer


def check_dependencies():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import websockets  # noqa: F401
    except ImportError as exc:
        python = shlex.quote(sys.executable)
        requirements = shlex.quote(str(Path(__file__).with_name("requirements.txt")))
        if importlib.util.find_spec("pip") is None:
            command = f"uv pip install --python {python} -r {requirements}"
        else:
            command = f"{python} -m pip install -r {requirements}"
        raise RuntimeError(f"Missing web dependency {exc.name!r}. Install into this Python environment:\n  {command}") from exc


def create_app(config, requests, responses, states, ready=None):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    # FastAPI resolves endpoint annotations against module globals.
    globals()["WebSocket"] = WebSocket
    clients = {}
    latest = None
    revision = 0

    async def pump():
        nonlocal latest, revision
        while True:
            for _ in range(64):
                try:
                    client, result = responses.get_nowait()
                except Empty:
                    break
                out = clients.get(client)
                if out is not None:
                    try:
                        out.put_nowait(result)
                    except asyncio.QueueFull:
                        # Closing a saturated client expires its control session.
                        offer(requests, (client, {"type": "release"}, time.monotonic()))
            try:
                latest = states.get_nowait()
                revision += 1
            except Empty:
                pass
            await asyncio.sleep(1/120)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(pump())
        if ready is not None:
            ready.set()
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/api/config")
    async def get_config():
        return config

    @app.websocket("/ws")
    async def socket(websocket: WebSocket):
        # This local dashboard accepts same-origin browser clients only.
        origin = websocket.headers.get("origin")
        if origin and origin != f"{websocket.url.scheme.replace('ws', 'http')}://{websocket.headers.get('host')}":
            await websocket.close(code=1008)
            return
        await websocket.accept()
        client = uuid.uuid4().hex
        out = asyncio.Queue(maxsize=16)
        clients[client] = out
        await websocket.send_json({"type": "hello", "client": client})

        async def sender():
            seen = -1
            while True:
                while not out.empty():
                    await asyncio.wait_for(websocket.send_json(out.get_nowait()), timeout=0.2)
                # Only send the newest state; never enqueue a stream per viewer.
                if latest is not None and revision != seen:
                    state = dict(latest)
                    age = max(0.0, time.monotonic()-state["time"])
                    state["state_age"] += age
                    state["fresh"] = state["fresh"] and state["state_age"] <= config["state_timeout"]
                    state["bridge_age"] = age
                    await asyncio.wait_for(websocket.send_json(state), timeout=0.2)
                    seen = revision
                await asyncio.sleep(1/30)

        async def receiver():
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 4096:
                    await websocket.close(code=1009)
                    return
                request_id = None
                try:
                    message = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    if not isinstance(message, dict):
                        raise ValueError("Expected an object")
                    request_id = message.get("id")
                    if request_id is not None and (not isinstance(request_id, int) or isinstance(request_id, bool)):
                        raise ValueError("Invalid request id")
                    if not offer(requests, (client, message, time.monotonic())):
                        raise ValueError("Controller busy; request rejected")
                except (ValueError, TypeError) as exc:
                    try:
                        valid_id = request_id if type(request_id) is int else None
                        out.put_nowait({"type": "result", "id": valid_id,
                                        "ok": False, "error": str(exc)})
                    except asyncio.QueueFull:
                        await websocket.close(code=1013)
                        return

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            clients.pop(client, None)
            offer(requests, (client, {"type": "release"}, time.monotonic()))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    dist = Path(__file__).parent / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist/"assets"), name="assets")
        app.mount("/robot", StaticFiles(directory=dist/"robot"), name="robot")
        app.mount("/licenses", StaticFiles(directory=dist/"licenses"), name="licenses")

        @app.get("/")
        async def index():
            return FileResponse(dist/"index.html")

    return app


def serve(config, requests, responses, states, ready, host, port):
    import socket
    import uvicorn
    # No reload and one worker: never instantiate a second robot controller.
    # Bind before reporting ready, including when another service owns the port.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.create_server((host, port), family=family) as listener:
        settings = uvicorn.Config(create_app(config, requests, responses, states, ready),
                                  host=host, port=port, log_level="warning",
                                  ws_max_size=4096, ws_max_queue=4)
        uvicorn.Server(settings).run(sockets=[listener])
