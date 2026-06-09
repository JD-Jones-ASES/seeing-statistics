r"""build_site.py — assemble the full static site into rendered/ for deployment.

This does NOT execute the notebooks (they are committed pre-executed and
fact-checked). It re-skins the pages from those executed notebooks and builds
every generated layer: landing page, data credits, glossary + formula sheet +
quiz data, the interactive explorables, the social card, the JupyterLite app,
and the Pagefind search index. The GitHub Actions workflow runs exactly this.

    ..\.venv\Scripts\python.exe tools\build_site.py

Run from anywhere; paths are resolved from this file. Requires (besides the
course's Python deps) Node/npx for Pagefind.
"""
import shutil
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable


def run(desc, args, optional=False):
    print(f"\n=== {desc} ===")
    try:
        subprocess.run(args, cwd=str(ROOT), check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if optional:
            print(f"  (skipped — {desc}: {e})")
            return False
        raise
    return True


def pagefind():
    print("\n=== Pagefind search index ===")
    site = str(ROOT / "rendered")
    cmds = [[PY, "-m", "pagefind", "--site", site]]   # pip-installed, if present
    npx = shutil.which("npx")                         # resolves npx.cmd on Windows
    if npx:
        cmds.append([npx, "-y", "pagefind@latest", "--site", site])
    pf = shutil.which("pagefind")
    if pf:
        cmds.append([pf, "--site", site])
    for cmd in cmds:
        try:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("  (skipped — pagefind not available; search index not built)")
    return False


def main():
    run("Re-export lesson pages (no re-execution)", [PY, "tools/render_html.py", "--no-exec"])
    run("Landing page", [PY, "tools/make_index.py"])
    run("Data sources & credits", [PY, "tools/make_credits.py"])
    if (ROOT / "tools" / "_reference.json").exists():
        run("Glossary + formula sheet + quizzes", [PY, "tools/make_reference.py"])
    else:
        print("\n=== Reference content === (skipped — tools/_reference.json absent)")
    run("Interactive explorables", [PY, "tools/build_explorables.py"])
    run("Scrollytelling story", [PY, "tools/make_scrolly.py"])
    run("Social card", [PY, "tools/make_og.py"], optional=True)   # needs matplotlib
    # Index search BEFORE building the JupyterLite app, and clear any stale copy,
    # so Pagefind only indexes course pages — not the Lite app's own HTML shells.
    live = ROOT / "rendered" / "live"
    if live.exists():
        shutil.rmtree(live)
    pagefind()
    run("JupyterLite app", [PY, "tools/build_jupyterlite.py"])
    print(f"\nSite assembled in {(ROOT / 'rendered')}")


if __name__ == "__main__":
    main()
