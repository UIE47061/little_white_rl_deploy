"""Bounded IPC; socket I/O and JSON serialization stay out of the control loop."""
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import threading
import time

from .control import WebControl


def offer(queue, value):
    try:
        queue.put_nowait(value)
        return True
    except Full:
        return False


class WebBridge:
    def __init__(self, controller, host="0.0.0.0", port=8080):
        from .server import serve, check_dependencies
        check_dependencies()
        if not (Path(__file__).parent / "dist/index.html").is_file():
            raise RuntimeError("Web assets missing. Run npm ci && npm run build in web/frontend")
        self.controller = controller
        self.control = WebControl(controller)
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=64)
        self.responses = context.Queue(maxsize=64)
        self.states = context.Queue(maxsize=1)
        self.ready = context.Event()
        self.process = context.Process(target=serve, name="little-white-web", daemon=True,
                                       args=(self.control.config, self.requests, self.responses,
                                             self.states, self.ready, host, port))
        self.closed = threading.Event()
        controller.web_control = self.control
        self.process.start()
        self.thread = threading.Thread(target=self.run, name="web-bridge", daemon=True)
        self.thread.start()
        deadline = time.monotonic()+10
        while not self.ready.wait(0.05):
            if not self.process.is_alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("Web server could not start; check dependencies and port")

    def run(self):
        next_state = 0.0
        try:
            while not self.closed.is_set():
                for _ in range(32):
                    try:
                        client, message, received = self.requests.get_nowait()
                    except Empty:
                        break
                    with self.controller.lock:
                        try:
                            self.control.handle(client, message, received)
                            result = {"type": "result", "id": message.get("id"), "ok": True}
                        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                            result = {"type": "result", "id": message.get("id"), "ok": False,
                                      "error": str(exc)}
                        # Successful replies carry the actual accepted values.
                        result["state"] = self.control.snapshot()
                    if message.get("id") is not None:
                        offer(self.responses, (client, result))
                now = time.monotonic()
                if now >= next_state and self.controller.lock.acquire(blocking=False):
                    try:
                        state = self.control.snapshot(now)
                    finally:
                        self.controller.lock.release()
                    offer(self.states, state)
                    next_state = now+1/30
                self.closed.wait(0.005)
        finally:
            with self.controller.lock:
                self.control.release("網頁服務已停止", time.monotonic())

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        for queue in (self.requests, self.responses, self.states):
            queue.cancel_join_thread()
            queue.close()
