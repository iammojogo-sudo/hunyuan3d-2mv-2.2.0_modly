"""Hunyuan3D-2mv — first-load bridge (v2.1.1).

This module is the SINGLE place where anything living OUTSIDE the extension
directory gets patched / bridged into shape. It runs inside the extension venv
(standard library only — no torch/trimesh needed, so it can repair those) and
is invoked:

  1. by generator.py on first load after install / weight download, and
  2. automatically by both generation bridges (shape + texture) right before
     the model loads, so any change that happened since the last run is
     bridged on the first generate.

What it does (all idempotent, all non-fatal):

  * bridge_weights()
      Model weights are downloaded by Modly's model-download step into the
      extension's node model dirs (models/hunyuan3d-2mv/generate, /texture).
      Depending on how the downloader / huggingface_hub laid them out, the
      subfolders hy3dgen needs (hunyuan3d-dit-v2-mv-turbo,
      hunyuan3d-paint-v2-0, hunyuan3d-delight-v2-0) may live:
        - under a SIBLING node dir (e.g. weights downloaded for the "texture"
          node while the user runs "generate"),
        - inside the hub's models--tencent--* cache (shared ~/.cache/huggingface
          or <model_dir>/.cache/huggingface/hub),
        - in the local-dir mirror <model_dir>/.cache/huggingface/download.
      bridge_weights() finds them wherever they are and hardlinks (fallback:
      copy) them into the layout hy3dgen's from_pretrained expects, so every
      run is fully offline. Hardlinks are instant and use zero extra disk.

  * repair_site_packages()
      The venv's site-packages lives outside the extension dir. If the bundled
      prebuilt custom_rasterizer CUDA kernel is missing there (or the venv was
      rebuilt), it is re-copied from the extension bundle so texture gen's
      `import custom_rasterizer` resolves with no compiler needed.

  * state file (ext_dir/.bridge_state.json)
      Records extension_version + what was bridged/repaired per node, so later
      runs (and generator.load()) are O(1) no-ops.

CLI:
    python hunyuan3d_bootstrap.py '<json_args>'

json_args:
    ext_dir    — absolute path to this extension directory
    model_dir  — active node's model dir (may be empty)
    node_id    — "generate" | "texture" | "texture_apply"; "" = repair-only
    siblings   — list of sibling node model dirs to search
    force      — skip the state fast-path and re-run everything (bool)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

EXTENSION_VERSION = "2.1.1"
STATE_FILE = ".bridge_state.json"

# subfolders each node needs, relative to that node's model dir
NODE_SUBFOLDERS = {
    "generate": ["hunyuan3d-dit-v2-mv-turbo"],
    "texture": ["hunyuan3d-paint-v2-0", "hunyuan3d-delight-v2-0"],
    "texture_apply": [],
}

# repos whose hub-cache layout we understand
HF_REPOS = ["models--tencent--Hunyuan3D-2mv", "models--tencent--Hunyuan3D-2"]

# a subfolder counts as "real" if it has any file that is not a huggingface
# hub .metadata marker
_METADATA_SUFFIXES = (".metadata",)


def _log(message):
    print(json.dumps({"type": "log", "message": str(message)}), flush=True)


def _subfolder_is_real(root, subfolder):
    """True when root/<subfolder> exists and contains actual weight files."""
    sub = Path(root) / subfolder
    if not sub.is_dir():
        return False
    try:
        for f in sub.rglob("*"):
            if f.is_file() and not f.name.endswith(_METADATA_SUFFIXES):
                return True
    except Exception:
        pass
    return False


def _hardlink_tree(src, dst):
    """Recursively hardlink src -> dst (fallback to copy2 on failure).

    Returns (linked, copied) file counts.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if f.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        if f.name.endswith(_METADATA_SUFFIXES):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == f.stat().st_size:
                continue
            try:
                target.unlink()
            except OSError:
                pass
        try:
            os.link(str(f), str(target))
            linked += 1
        except OSError:
            try:
                shutil.copy2(str(f), str(target))
                copied += 1
            except OSError as e:
                _log(f"[bridge] copy failed {rel}: {e}")
    return linked, copied


# --------------------------------------------------------------------------- #
# Weight discovery                                                             #
# --------------------------------------------------------------------------- #
def _hub_cache_subfolder_candidates(model_dir, subfolder):
    """Return (source_path, label) candidates for `subfolder` found inside the
    huggingface hub caches (shared user cache or the local-dir cache)."""
    candidates = []
    roots = []
    home_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if home_hub.is_dir():
        roots.append((home_hub, "~/.cache/huggingface/hub"))
    if model_dir:
        local_hub = Path(model_dir) / ".cache" / "huggingface" / "hub"
        if local_hub.is_dir():
            roots.append((local_hub, "<model_dir>/.cache/huggingface/hub"))
        mirror = Path(model_dir) / ".cache" / "huggingface" / "download"
        if mirror.is_dir() and _subfolder_is_real(mirror, subfolder):
            candidates.append((str(mirror / subfolder), "local-dir mirror (.cache/huggingface/download)"))
    for root, label in roots:
        for repo in HF_REPOS:
            repo_dir = root / repo
            if not repo_dir.is_dir():
                continue
            snap = repo_dir / "snapshots"
            if not snap.is_dir():
                continue
            # prefer the most recently modified snapshot with our subfolder
            found = [d for d in snap.iterdir() if d.is_dir() and _subfolder_is_real(d, subfolder)]
            if not found:
                continue
            found.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            candidates.append((str(found[0] / subfolder), f"hf hub cache ({repo})"))
    return candidates


def _find_subfolder_sources(model_dir, siblings, subfolder):
    """Return a list of (source_path, label) where `subfolder` weights live."""
    sources = []
    if model_dir:
        if _subfolder_is_real(model_dir, subfolder):
            sources.append((str(Path(model_dir) / subfolder), "active node dir"))
    for sib in siblings or []:
        if _subfolder_is_real(sib, subfolder):
            label = f"sibling node dir ({os.path.basename(sib)})"
            sources.append((str(Path(sib) / subfolder), label))
    seen = {str(Path(s[0]).resolve()) for s in sources}
    for src, label in _hub_cache_subfolder_candidates(model_dir, subfolder):
        if str(Path(src).resolve()) not in seen:
            seen.add(str(Path(src).resolve()))
            sources.append((src, label))
    return sources


def bridge_weights(model_dir, siblings, node_id):
    """Ensure every subfolder the active node needs exists under model_dir.

    Missing subfolders are hardlinked (fallback copy) from the first source
    found (sibling node dirs, hub cache, local-dir mirror). Returns a dict
    {subfolder: {"ok": bool, "bridged": bool, "source": str}} and logs what it
    did. Missing-with-no-source stays "ok": False so the caller surfaces a
    clear "download the weights via Modly's model step" error.
    """
    if not model_dir:
        _log("[bridge] no model_dir provided — weight bridging skipped")
        return {}
    specs = NODE_SUBFOLDERS.get(str(node_id or "").lower())
    if not specs:
        specs = []
        for subs in NODE_SUBFOLDERS.values():
            for s in subs:
                if s not in specs:
                    specs.append(s)
    model_dir = str(Path(model_dir).resolve())
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    status = {}
    for sub in specs:
        entry = {"ok": False, "bridged": False, "source": ""}
        if _subfolder_is_real(model_dir, sub):
            entry["ok"] = True
            entry["source"] = "active node dir"
            status[sub] = entry
            continue
        sources = _find_subfolder_sources(model_dir, siblings, sub)
        if sources:
            src, label = sources[0]
            if Path(src).resolve() == Path(model_dir).resolve() / sub:
                entry["ok"] = True
                entry["source"] = label
            else:
                linked, copied = _hardlink_tree(src, str(Path(model_dir) / sub))
                if _subfolder_is_real(model_dir, sub):
                    entry["ok"] = True
                    entry["bridged"] = True
                    entry["source"] = label
                    _log(f"[bridge] bridged {sub} -> {Path(model_dir) / sub} "
                         f"from {label} ({linked} linked, {copied} copied)")
                else:
                    _log(f"[bridge] failed to bridge {sub} from {label}")
        else:
            _log(f"[bridge] {sub} not found in active dir, siblings, or HF cache — "
                 "download it via Modly's model-download step for this node")
        status[sub] = entry
    return status


# --------------------------------------------------------------------------- #
# site-packages repair (out-of-extension files)                               #
# --------------------------------------------------------------------------- #
def _site_packages(ext_dir):
    venv = Path(ext_dir) / "venv"
    sp = venv / "Lib" / "site-packages"
    if sp.is_dir():
        return sp
    matches = sorted((venv / "lib").glob("python*/site-packages")) if (venv / "lib").is_dir() else []
    return matches[-1] if matches else None


def _kernel_import_ok():
    """custom_rasterizer_kernel requires torch's DLLs on the search path, so
    import torch first (exactly like hy3dgen does at runtime).

    Returns (ok, error_message_or_None)."""
    try:
        import torch  # noqa: F401
        import custom_rasterizer  # noqa: F401
        import custom_rasterizer_kernel  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def repair_site_packages(ext_dir):
    """Verify / repair the venv's custom_rasterizer install (lives outside the
    extension dir). Returns True when healthy."""
    bundle = Path(ext_dir) / "custom_rasterizer"
    sp = _site_packages(ext_dir)
    if sp is None:
        _log("[bridge] venv site-packages not found — can't repair")
        return False
    ok, err = _kernel_import_ok()
    if ok:
        return True
    _log(f"[bridge] custom_rasterizer import check failed: {err}")
    if not bundle.is_dir():
        _log("[bridge] custom_rasterizer bundle missing from extension dir — "
             "texture gen needs it (re-download the extension)")
        return False
    dest = sp / "custom_rasterizer"
    try:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(str(bundle), str(dest))
        pth = sp / "hy3d_custom_rasterizer.pth"
        pth.write_text(str(dest) + "\n", encoding="utf-8")
        _log(f"[bridge] re-installed custom_rasterizer bundle -> {dest}")
    except Exception as e:
        _log(f"[bridge] custom_rasterizer repair failed: {e}")
        return False
    ok, err = _kernel_import_ok()
    if ok:
        _log("[bridge] custom_rasterizer import OK after repair")
        return True
    _log(f"[bridge] custom_rasterizer still failing to import: {err}")
    return False


# --------------------------------------------------------------------------- #
# orchestrator                                                                #
# --------------------------------------------------------------------------- #
def ensure_bridged(args):
    """Run the full first-load bridge. Idempotent; fast-path via state file."""
    ext_dir = str(Path(args.get("ext_dir") or Path(__file__).resolve().parent).resolve())
    model_dir = str(Path(args.get("model_dir") or "").resolve()) if args.get("model_dir") else ""
    node_id = str(args.get("node_id", "")).lower()
    siblings = [str(Path(s).resolve()) for s in (args.get("siblings") or []) if s]
    force = bool(args.get("force"))

    state_path = Path(ext_dir) / STATE_FILE
    state = {}
    if state_path.exists() and not force:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    if not isinstance(state, dict):
        state = {}

    if state.get("extension_version") == EXTENSION_VERSION and not force:
        # Fast path: everything was already bridged for this build.
        _log(f"[bridge] already bridged (v{EXTENSION_VERSION}) — nothing to do")
        return state

    repaired = repair_site_packages(ext_dir)

    status = {}
    if node_id in NODE_SUBFOLDERS:
        status[node_id] = bridge_weights(model_dir, siblings, node_id)
    else:
        # No active node id (e.g. first load before any generate) — repair
        # out-of-extension files only. Per-node weight bridging happens inside
        # the generation bridges on the first generate, when the node is known.
        _log("[bridge] no node_id — weight bridging deferred to first generate")

    state = {
        "extension_version": EXTENSION_VERSION,
        "ts": int(time.time()),
        "site_packages_repaired": repaired,
        "nodes": status,
    }
    try:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _log(f"[bridge] state written -> {state_path}")
    except Exception as e:
        _log(f"[bridge] failed to write state file: {e}")
    return state


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    raw = argv[1] if len(argv) > 1 else "{}"
    try:
        args = json.loads(raw)
    except Exception:
        _log(f"[bridge] bad JSON args: {raw}")
        sys.exit(1)
    try:
        ensure_bridged(args)
    except Exception as e:
        _log(f"[bridge] unexpected failure: {e}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
