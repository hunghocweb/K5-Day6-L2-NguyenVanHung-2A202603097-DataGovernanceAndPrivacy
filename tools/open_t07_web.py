"""Double-click launcher: reuse/start the local server, then open its URL."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def ready(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/api/state", timeout=1) as response:
            state = json.load(response)
        return (state.get("gate", {}).get("scope") == "t07_web_review"
                and isinstance(state.get("items"), list) and bool(state.get("token")))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def launch(port=8765, open_browser=True, page="/"):
    url = f"http://127.0.0.1:{port}"
    if not ready(url):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Cong {port} dang duoc ung dung khac su dung. Dong ung dung do roi thu lai.")
        output = ROOT / "out/web_review"
        output.mkdir(parents=True, exist_ok=True)
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        with (output / "server.stdout.log").open("ab") as stdout, (output / "server.stderr.log").open("ab") as stderr:
            process = subprocess.Popen([sys.executable, "-m", "tools.t07_review_web", "--port", str(port)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=flags, start_new_session=os.name != "nt")
        (output / "server.pid").write_text(str(process.pid), encoding="ascii")
        deadline = time.monotonic() + 30
        while not ready(url):
            if process.poll() is not None:
                raise RuntimeError("Server khong khoi dong duoc. Xem out/web_review/server.stderr.log")
            if time.monotonic() >= deadline:
                raise RuntimeError("Server khoi dong cham. Xem out/web_review/server.stderr.log va thu mo lai.")
            time.sleep(.25)
    target=url+page
    if open_browser and not webbrowser.open(target, new=2):
        print(f"Hay mo trinh duyet va truy cap: {target}")
    else:
        print(f"T07 web ready: {target}")
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Verify startup without opening a browser")
    parser.add_argument("--page", choices=["/", "/walkthrough"],default="/")
    args = parser.parse_args()
    try:
        launch(args.port, not args.no_browser,args.page)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
