"""
run_dashboard.py — Launch the FireGuard AI Web Dashboard
─────────────────────────────────────────────────────────
Usage:
    python run_dashboard.py
    python run_dashboard.py --port 5000
    python run_dashboard.py --host 0.0.0.0 --port 8080

Opens browser automatically at http://localhost:5000
Default login: admin / admin123
"""

import argparse
import sys
import os
import webbrowser
import threading
from pathlib import Path

# Add frontend and backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))


def main():
    parser = argparse.ArgumentParser(description="FireGuard AI Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    from dashboard.app import socketio, app

    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"

    print("\n" + "=" * 60)
    print("  [*] FireGuard AI Dashboard")
    print(f"  URL:      {url}")
    print(f"  Login:    admin / admin123")
    print(f"  Host:     {args.host}:{args.port}")
    print("=" * 60 + "\n")

    if not args.no_browser:
        # Open browser after 1.5s to allow server to start
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
