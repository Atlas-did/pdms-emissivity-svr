from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _pick_port(start: int = 8501, max_tries: int = 30) -> int:
    port = start
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("无法分配可用端口，请关闭占用 8501~8530 的进程后重试。")


def _resolve_web_ui_file() -> Path:
    candidates = []

    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            meipass / "stage1" / "web_ui.py",
            exe_dir / "stage1" / "web_ui.py",
        ])

    here = Path(__file__).resolve().parents[1]
    candidates.append(here / "stage1" / "web_ui.py")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError("未找到 stage1/web_ui.py，请检查打包时是否包含该文件。")


def _open_browser_later(url: str, delay: float = 2.0) -> None:
    time.sleep(delay)
    webbrowser.open(url)


def main() -> int:
    port = _pick_port(8501)
    url = f"http://127.0.0.1:{port}"

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))

    app_file = _resolve_web_ui_file()

    threading.Thread(target=_open_browser_later, args=(url, 1.8), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_file),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    return int(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
