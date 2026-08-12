#!/usr/bin/env python3
"""One-command setup + launch for all four VitalHealth processes.

Creates each app's venv and installs its requirements.txt if missing, then
starts Jace, Jeslyn, and YS (in that order), followed by the gateway once
all three backends are reachable, and opens the gateway in a browser.

This script is tooling only — it contains no business logic and does not
import or modify any app. Stdlib only, so it can run before any venv exists.

Stop everything with Ctrl+C. Shutdown tree-kills each process (not just its
top-level PID) because Flask's debug reloader (used by Jeslyn) spawns a
child process that survives a plain terminate() and orphans itself on its
port — confirmed the hard way during development of this script.
"""
import os
import hashlib
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure our own progress output appears immediately even when redirected to
# a file or pipe (Python fully buffers stdout in that case by default,
# unlike an interactive terminal, which is line-buffered).
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform.startswith("win")
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 12)
STARTUP_TIMEOUT_SECONDS = 120.0

# Order matters: backends first, gateway last.
APPS = [
    {
        "name": "Jace",
        "dir": ROOT / "apps" / "Jace",
        "cmd": ["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        "port": 8000,
        "extra_env": {},
    },
    {
        "name": "Jeslyn",
        "dir": ROOT / "apps" / "Jeslyn",
        "cmd": ["python", "app.py"],
        "port": 5000,
        "extra_env": {},
    },
    {
        "name": "YS",
        "dir": ROOT / "apps" / "YS",
        "cmd": ["python", "app.py"],
        "port": 5001,
        "extra_env": {"PORT": "5001"},
    },
    {
        "name": "gateway",
        "dir": ROOT / "apps" / "gateway",
        "cmd": ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8080"],
        "port": 8080,
        "extra_env": {},
    },
]

GATEWAY_URL = "http://127.0.0.1:8080/"


def venv_python(app_dir: Path) -> Path:
    if IS_WINDOWS:
        return app_dir / ".venv" / "Scripts" / "python.exe"
    return app_dir / ".venv" / "bin" / "python"


def supported_python(version: tuple[int, int]) -> bool:
    return MIN_PYTHON <= version <= MAX_PYTHON


def require_supported_launcher_python():
    version = sys.version_info[:2]
    if supported_python(version):
        return
    print(
        "VitalHealth requires Python 3.11 or 3.12. "
        f"This launcher is running under Python {version[0]}.{version[1]}.\n"
        "Install Python 3.12, then run: py -3.12 run_all.py",
        file=sys.stderr,
    )
    raise SystemExit(1)


def venv_version(py: Path) -> tuple[int, int] | None:
    """Read a venv interpreter's major/minor version without importing apps."""
    result = subprocess.run(
        [str(py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.strip().split(".", maxsplit=1)
        return int(major), int(minor)
    except ValueError:
        return None


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def load_env_file(path: Path) -> dict:
    """Minimal KEY=VALUE .env parser — no third-party dependency needed
    here since this script must run before any venv (and thus python-dotenv)
    exists."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def preflight_check_ports():
    conflicts = [(app["name"], app["port"]) for app in APPS if port_in_use(app["port"])]
    if conflicts:
        print("Cannot start - the following ports are already in use:")
        for name, port in conflicts:
            print(f"  - {name} wants port {port}, but something is already listening there.")
        print("\nStop whatever is using those ports (see README troubleshooting) and re-run.")
        sys.exit(1)


def ensure_app_ready(app: dict):
    app_dir = app["dir"]
    py = venv_python(app_dir)
    if py.exists():
        version = venv_version(py)
        if version and supported_python(version):
            ensure_requirements_current(app, py)
            ensure_shared_storage(app, py)
            return
        raise RuntimeError(
            f"{app['name']} has an incompatible or broken virtual environment at {py.parent.parent}. "
            "Delete that .venv and re-run this script with Python 3.12."
        )
    print(f"[{app['name']}] no venv found - setting up (first run only, may take a few minutes)...")
    subprocess.run([sys.executable, "-m", "venv", ".venv"], cwd=app_dir, check=True)
    install_requirements(app, py)
    ensure_shared_storage(app, py)
    print(f"[{app['name']}] setup complete.")


def requirements_fingerprint(app_dir: Path) -> str:
    """Track dependency changes so existing venvs are upgraded once."""
    digest = hashlib.sha256()
    for path in (app_dir / "requirements.txt", ROOT / "pyproject.toml"):
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def install_requirements(app: dict, py: Path):
    subprocess.run(
        [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"],
        cwd=app["dir"],
        check=True,
    )
    (py.parent.parent / ".vitalhealth_requirements.sha256").write_text(
        requirements_fingerprint(app["dir"]), encoding="utf-8"
    )


def ensure_requirements_current(app: dict, py: Path):
    stamp = py.parent.parent / ".vitalhealth_requirements.sha256"
    expected = requirements_fingerprint(app["dir"])
    current = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
    if current == expected:
        return
    print(f"[{app['name']}] requirements changed - updating its virtual environment...")
    install_requirements(app, py)


def ensure_shared_storage(app: dict, py: Path):
    """Upgrade pre-existing environments after shared code is added.

    Each app's requirements.txt installs this package on a fresh setup
    (the gateway needs it for the `users` table backing login; the three
    backends need it for clinical records); this small check also repairs
    virtual environments created before the package was added to that app.
    """
    probe = subprocess.run(
        [str(py), "-c", "import vitalhealth_storage"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        return
    print(f"[{app['name']}] installing shared database package...")
    subprocess.run(
        [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-e", str(ROOT)],
        check=True,
    )


def stream_output(name: str, proc: subprocess.Popen):
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        print(f"[{name}] {line.rstrip()}")


def start_app(app: dict) -> subprocess.Popen:
    py = venv_python(app["dir"])
    program, *args = app["cmd"]
    if program == "uvicorn":
        cmd = [str(py), "-m", "uvicorn", *args]
    else:  # "python"
        cmd = [str(py), *args]

    env = os.environ.copy()
    env.update(load_env_file(app["dir"] / ".env"))

    if app["name"] == "gateway":
        # The gateway owns login (shared DATABASE_URL) and Vita (Gemini), but
        # each clinical app keeps its provider credentials in its own .env.
        # Reuse Jace's local settings only when the gateway does not define an
        # explicit value itself. This keeps one-command local startup working
        # after the login/Vita UI was added without duplicating secrets.
        jace_env = load_env_file(ROOT / "apps" / "Jace" / ".env")
        if not env.get("DATABASE_URL", "").strip():
            env["DATABASE_URL"] = jace_env.get("DATABASE_URL", "")
        if not env.get("GEMINI_API_KEY", "").strip():
            env["GEMINI_API_KEY"] = jace_env.get("GEMINI_API_KEY", "")

    env.update(app["extra_env"])

    popen_kwargs = dict(
        cwd=app["dir"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    threading.Thread(target=stream_output, args=(app["name"], proc), daemon=True).start()
    return proc


def kill_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    require_supported_launcher_python()
    preflight_check_ports()

    try:
        for app in APPS:
            ensure_app_ready(app)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    procs = []
    try:
        for app in APPS:
            print(f"[{app['name']}] starting on port {app['port']}...")
            proc = start_app(app)
            procs.append(proc)

            if not wait_for_port(app["port"]):
                exit_code = proc.poll()
                print(f"[{app['name']}] failed to come up on port {app['port']} "
                      f"(exit code: {exit_code}). See output above.")
                raise SystemExit(1)
            print(f"[{app['name']}] up.")

        print(f"\nAll four processes are up. Opening {GATEWAY_URL}")
        webbrowser.open(GATEWAY_URL)

        print("Press Ctrl+C to stop everything.\n")
        while True:
            time.sleep(1)
            for app, proc in zip(APPS, procs):
                if proc.poll() is not None:
                    print(f"[{app['name']}] exited unexpectedly (code {proc.returncode}).")
                    raise SystemExit(1)

    except KeyboardInterrupt:
        print("\nStopping all processes...")
    finally:
        for proc in procs:
            kill_tree(proc)
        print("Stopped.")


if __name__ == "__main__":
    main()
