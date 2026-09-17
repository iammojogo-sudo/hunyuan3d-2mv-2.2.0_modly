import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress

import json as _modly_json
import time as _modly_time
import uuid as _modly_uuid
from pathlib import Path as _modly_Path


def _modly_session_file(outputs_dir):
    return _modly_Path(outputs_dir) / ".modly_run.json"


def _modly_new_run_folder(outputs_dir):
    outputs_dir = _modly_Path(outputs_dir)
    run = outputs_dir / f"run_{int(_modly_time.time())}_{_modly_uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=True)
    _modly_session_file(outputs_dir).write_text(
        _modly_json.dumps({"run_folder": str(run)}), encoding="utf-8")
    return run


def _modly_current_run_folder(outputs_dir, params=None, input_path=None):
    outputs_dir = _modly_Path(outputs_dir)
    params = params or {}
    rf = params.get("run_folder") or ""
    if rf and _modly_Path(rf).is_dir():
        return _modly_Path(rf)
    # Check multiple param keys that may carry the upstream node's output path.
    # mesh_path is critical for the texture node (input mesh lives in the
    # generate run folder). We check ALL candidate keys and prefer the first
    # one whose path hierarchy contains a run_ folder — this avoids picking
    # an image_path that lives outside any run folder while the mesh is inside one.
    _candidate_srcs = []
    for k in ("input_path", "mesh_path", "image_path"):
        v = params.get(k)
        if v:
            _candidate_srcs.append(v)
    if input_path:
        _candidate_srcs.append(input_path)
    for src in _candidate_srcs:
        if src:
            p = _modly_Path(src)
            for cand in ([p] + list(p.parents)):
                if cand.is_dir() and cand.name.startswith("run_"):
                    return cand
            # If the path is relative (e.g. "Workflows/run_xxx/mesh.glb"),
            # the above check failed because it resolved against the wrong
            # CWD.  Try resolving against the parent of outputs_dir (workspace
            # root) to find the actual run folder.
            if not p.is_absolute():
                _resolved = _modly_Path(outputs_dir).parent / str(p)
                for cand in ([_resolved] + list(_resolved.parents)):
                    if cand.is_dir() and cand.name.startswith("run_"):
                        return cand
    # If mesh_path was explicitly provided by the user and doesn't reside in any
    # run_ folder, create a fresh run folder instead of reusing a stale one from
    # the session file (which would belong to a previous unrelated generation).
    if params.get("mesh_path"):
        return _modly_new_run_folder(outputs_dir)
    # Session file: tracks the most recent run created in this Python session.
    sf = _modly_session_file(outputs_dir)
    if sf.exists():
        try:
            data = _modly_json.loads(sf.read_text(encoding="utf-8"))
            p = _modly_Path(data.get("run_folder", ""))
            if p.is_dir():
                return p
        except Exception:
            pass
    # Nothing found — create a brand new run folder instead of recycling an old
    # one. This avoids mixing outputs from different generations.
    return _modly_new_run_folder(outputs_dir)


EXT_DIR = Path(__file__).resolve().parent

_RENDER_VIEWS = [
    ("front",  270,   0),
    ("right",    0,   0),
    ("back",    90,   0),
    ("left",   180,   0),
    ("top",      0,  90),
    ("bottom",   0, -90),
]


def _look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    fwd = np.array(target) - np.array(eye)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array(up)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat


def _render_mesh_views(mesh_path, output_dir):
    import trimesh
    import pyrender
    from PIL import Image as _PIL

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        print("Warning: mesh has no vertices, skipping view rendering", file=sys.stderr, flush=True)
        return None, None
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    verts = np.array(mesh.vertices, copy=True)
    verts -= center
    scale = np.max(bounds[1] - bounds[0])
    if scale > 0:
        verts /= scale
    verts = verts @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    mesh.vertices = verts

    viz = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    views_dir = output_dir / "rendered_views"
    views_dir.mkdir(exist_ok=True)
    depths_dir = output_dir / "rendered_depths"
    depths_dir.mkdir(exist_ok=True)

    bg = np.array([0.5, 0.5, 0.5])
    dist = 1.5
    view_order = ["front", "left", "back", "right", "top", "bottom"]
    rendered = {}
    rendered_depth = {}
    for name, azim, elev in _RENDER_VIEWS:
        scene = pyrender.Scene(bg_color=bg, ambient_light=[0.3, 0.3, 0.3])
        scene.add(viz)
        azim_rad = math.radians(azim)
        elev_rad = math.radians(elev)
        eye = [
            dist * math.cos(elev_rad) * math.cos(azim_rad),
            dist * math.cos(elev_rad) * math.sin(azim_rad),
            dist * math.sin(elev_rad),
        ]
        cam_pose = _look_at(eye)
        scene.add(pyrender.OrthographicCamera(xmag=1.1, ymag=1.1, znear=0.01, zfar=dist * 2), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose)
        opp_eye = [-x for x in eye]
        opp_pose = _look_at(opp_eye)
        scene.add(pyrender.DirectionalLight(color=[0.6, 0.6, 0.6], intensity=1.5), pose=opp_pose)
        r = pyrender.OffscreenRenderer(512, 512)
        color, depth = r.render(scene)
        r.delete()
        _PIL.fromarray(color).save(str(views_dir / f"{name}.png"))
        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth_norm = (depth - dmin) / (dmax - dmin)
        else:
            depth_norm = depth
        depth_out = np.clip(depth_norm * 65535, 0, 65535).astype(np.uint16)
        _PIL.fromarray(depth_out).save(str(depths_dir / f"{name}.png"))
        rendered[name] = _PIL.fromarray(color)
        rendered_depth[name] = _PIL.fromarray(depth_out)

    tile_w = 512 * 2
    tile_h = 512 * 3
    tile = _PIL.new("RGB", (tile_w, tile_h))
    positions = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    for i, name in enumerate(view_order):
        img = rendered.get(name)
        if img is not None:
            col, row = positions[i]
            tile.paste(img, (col * 512, row * 512))
    tile_path = output_dir / "rendered_tiled.png"
    tile.save(str(tile_path))

    return views_dir, depths_dir


def _rembg_single(in_path, out_path):
    """Run rembg via venv on a single image file, writing the RGBA result."""
    import subprocess as _sp
    _venv = str(EXT_DIR / "venv" / "Scripts" / "python.exe")
    _in = in_path.replace("\\", "/")
    _out = out_path.replace("\\", "/")
    _code = (
        "from rembg import remove, new_session;"
        "from PIL import Image;"
        f"i = Image.open(r'{_in}').convert('RGB');"
        f"o = remove(i, session=new_session(providers=['CPUExecutionProvider']), bgcolor=[255,255,255,0]);"
        f"o.save(r'{_out}')"
    )
    try:
        _sp.run([_venv, "-c", _code], capture_output=True, timeout=300, check=True)
        return True
    except Exception as e:
        print(f"[hunyuan3d] rembg failed: {e}", file=sys.stderr, flush=True)
        return False


def _auto_tile_folder(folder_path, output_dir, shape_views=4):
    """Scan a folder for images, arrange into a grid (2x2 for ≤4 views,
    2x3 for 5-6 views), save the composite to output_dir/folder_tiled.png,
    and return its Path.

    Filenames containing 'front', 'left', 'back', 'right', 'top', 'bottom'
    are placed in their matching grid position. Unrecognised names fill
    remaining slots in alphabetical order.
    """
    from PIL import Image as _PIL
    import glob
    import re

    _exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    _files = []
    for ext in _exts:
        _files.extend(glob.glob(os.path.join(folder_path, ext)))
        _files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    _files = sorted(set(_files))
    if not _files:
        raise RuntimeError(f"No supported images found in folder: {folder_path}")

    # Match filenames to grid positions by view keyword
    _view_kw = {"front": 0, "left": 1, "back": 2, "right": 3, "top": 4, "bottom": 5}
    _max_slots = 6 if shape_views > 4 else 4
    _slots = [None] * _max_slots
    _unmatched = []
    for f in _files:
        _name = os.path.splitext(os.path.basename(f))[0].lower()
        _name = _name.replace("_", " ").replace("-", " ")
        _found = False
        for kw, idx in _view_kw.items():
            if re.search(r'\b' + kw + r'\b', _name):
                if _slots[idx] is None:
                    _slots[idx] = f
                    _found = True
                    break
        if not _found:
            _unmatched.append(f)

    _ordered = []
    for s in _slots:
        if s is not None:
            _ordered.append(s)
        elif _unmatched:
            _ordered.append(_unmatched.pop(0))
    # Remove any None slots left when _unmatched ran out.
    _ordered = [f for f in _ordered if f is not None][:_max_slots]

    if not _ordered:
        raise RuntimeError(f"No images could be assigned to grid slots in folder: {folder_path}")

    imgs = [_PIL.open(f).convert("RGBA") for f in _ordered]

    # ── Step 1: Remove backgrounds from each original image ──────────────
    # Uses the venv's rembg (not available in host Python). Saves as
    # rembg_<original_name>.png in the source folder for the user to reuse.
    import subprocess as _sp
    _venv_py = str(EXT_DIR / "venv" / "Scripts" / "python.exe")
    _rm_script = []
    _rm_script.append("import sys, os")
    _rm_src = {}  # _i -> path to rembg output
    for _i, _f in enumerate(_ordered):
        _stem = os.path.splitext(os.path.basename(_f))[0]
        _out = os.path.join(str(output_dir), f"rembg_{_stem}.png")
        _rm_src[_i] = _out
        # Build the inline command (use forward slashes to avoid any escaping issues)
        _in_p  = _f.replace("\\", "/")
        _out_p = _out.replace("\\", "/")
        _rm_script.append(
            f"from rembg import remove, new_session;"
            f"from PIL import Image;"
            f"i = Image.open(r'{_in_p}').convert('RGB');"
            f"o = remove(i, session=new_session(), bgcolor=[255,255,255,0]);"
            f"o.save(r'{_out_p}')"
        )
    try:
        _sp.run([_venv_py, "-c", "; ".join(_rm_script)],
                capture_output=True, timeout=300, check=True)
        # Reload the rembg'd images
        for _i in range(len(imgs)):
            imgs[_i] = _PIL.open(_rm_src[_i]).convert("RGBA")
        print(f"[hunyuan3d] Backgrounds removed -> rembg_*.png saved in {output_dir}",
              file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[hunyuan3d] Background removal failed: {_e}", file=sys.stderr, flush=True)
        # Fall through with the original images

    # ── Step 2: Pad each image with TRANSPARENT margins to equal size ────
    # Choose grid layout first, then size cells so the tile's aspect ratio
    # matches what the bridge's _detect_grid will pick (closest to nc/nr).
    if shape_views > 4:
        cols, rows = 2, 3
    else:
        cols, rows = 2, 2
    _raw_max_w = max(i.width for i in imgs)
    _raw_max_h = max(i.height for i in imgs)
    # Cells are square (cw == ch).  Tile aspect = cols/rows, which is what
    # _detect_grid expects (1.0 for 2×2, 0.667 for 2×3, 1.5 for 3×2).
    _cell = max(_raw_max_w, _raw_max_h)
    _cw = _ch = _cell

    _padded = []
    for _i, img in enumerate(imgs):
        canvas = _PIL.new("RGBA", (_cw, _ch), (0, 0, 0, 0))  # fully transparent
        # Scale down if larger than cell, keeping aspect ratio
        _scale = min(_cw / img.width, _ch / img.height, 1.0)
        _nw = int(img.width * _scale)
        _nh = int(img.height * _scale)
        _resized = img.resize((_nw, _nh), _PIL.LANCZOS)
        x = (_cw - _nw) // 2
        y = (_ch - _nh) // 2
        canvas.paste(_resized, (x, y), _resized)
        _padded.append(canvas)
        _stem = os.path.splitext(os.path.basename(_ordered[_i]))[0]
        _padded_path = os.path.join(str(output_dir), f"padded_{_stem}.png")
        canvas.save(_padded_path)

    # ── Step 3: Tile the padded images into a grid ─────────────────────
    tile = _PIL.new("RGBA", (_cw * cols, _ch * rows), (0, 0, 0, 0))  # transparent bg
    _positions = [(c, r) for r in range(rows) for c in range(cols)]
    for idx, img in enumerate(_padded):
        if idx < len(_positions):
            col, row = _positions[idx]
            tile.paste(img, (col * _cw, row * _ch), img)

    tile_path = output_dir / "folder_tiled.png"
    tile.save(str(tile_path))
    print(f"[hunyuan3d] Auto-tiled {len(imgs)} image(s) from {folder_path} -> {tile_path}",
          file=sys.stderr, flush=True)
    return tile_path


def _venv_python() -> Path:
    is_win = platform.system() == "Windows"
    return EXT_DIR / "venv" / ("Scripts/python.exe" if is_win else "bin/python")


class Hunyuan3DmvGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d-2mv"
    DISPLAY_NAME = "Hunyuan3D-2mv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.python_path = None
        self.hy3dgen_path = ""
        self.model_cache = None

    def load(self) -> None:
        self.python_path = _venv_python()
        if not self.python_path.exists():
            raise RuntimeError(
                "Hunyuan3D-2mv venv not found. Re-install the extension so "
                "setup.py can build it (creates venv/ with torch + hy3dgen)."
            )
        # hy3dgen is pip-installed into the venv; no extra sys.path needed.
        self.hy3dgen_path = ""
        # Weights are read from Modly's local model dir (offline).
        self.model_cache = self.model_dir
        # Bridge anything Modly placed OUTSIDE the extension (weights in the
        # models/ dir, venv site-packages) on first load after install/download.
        self._bridge_on_load()

    def _bridge_on_load(self) -> None:
        """Run the first-load bridge (hunyuan3d_bootstrap.py).

        After Modly downloads the node weights (or after an update), the
        weights / venv files may not be in the layout hy3dgen expects yet.
        The bridge locates them wherever Modly put them (sibling node dirs,
        HF hub cache) and hardlinks them into place, and repairs the venv's
        custom_rasterizer if needed. Fast-path: when .bridge_state.json is
        already fresh for v2.2.0 AND every known subfolder is present, this is
        a no-op (no subprocess). Never raises — failures are logged.
        """
        bootstrap = EXT_DIR / "hunyuan3d_bootstrap.py"
        if not bootstrap.exists():
            return

        siblings = []
        parent = self.model_dir.parent
        if parent.exists():
            siblings = [p for p in sorted(parent.iterdir()) if p.is_dir() and p != self.model_dir]

        _specs = [
            ["hunyuan3d-dit-v2-mv-turbo"],
            ["hunyuan3d-paint-v2-0", "hunyuan3d-delight-v2-0"],
        ]
        _roots = [self.model_dir] + siblings

        def _layout_ready() -> bool:
            for subs in _specs:
                for s in subs:
                    if not any((r / s).is_dir() for r in _roots):
                        return False
            return True

        state_file = EXT_DIR / ".bridge_state.json"
        if state_file.exists():
            try:
                _state = json.loads(state_file.read_text(encoding="utf-8"))
                if (_state.get("extension_version") == "2.2.0"
                        and _state.get("site_packages_repaired")
                        and _layout_ready()):
                    return
            except Exception:
                pass

        args = {
            "ext_dir": str(EXT_DIR),
            "model_dir": str(self.model_dir),
            "node_id": str(getattr(self, "node_id", "") or "").lower(),
            "siblings": [str(s) for s in siblings],
        }
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[{self.MODEL_ID}] Running first-load bridge…", file=sys.stderr, flush=True)
        try:
            proc = subprocess.run(
                [str(self.python_path), str(bootstrap), json.dumps(args)],
                capture_output=True, text=True, timeout=600, env=env,
            )
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    text = msg.get("message") or line
                except Exception:
                    text = line
                print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[{self.MODEL_ID}] First-load bridge failed (non-fatal): {e}",
                  file=sys.stderr, flush=True)

    def is_downloaded(self) -> bool:
        # Weights live in Modly's local model dir (downloaded by setup.py or
        # the manifest-driven _auto_download). Check there so Modly reports
        # the correct status and never re-fetches from the network.
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def unload(self) -> None:
        super().unload()

    def _ensure_shape_weights(self) -> str:
        """Ensure the shape (DIT-mv) weights exist in Modly's local model dir
        (or a sibling node dir where Modly actually placed them). No network
        re-fetch — surface a clear error if missing.

        Returns the RESOLVED root dir containing the shape subfolder (so the
        bridge passes the correct path even when the active node's model_dir is
        a different node, e.g. 'texture')."""
        sub = "hunyuan3d-dit-v2-mv-turbo"
        root = self._find_weights_root(sub)
        if root is not None:
            return str(root)
        raise RuntimeError(
            f"Shape weights ({sub}) not found under {self.model_dir} or its "
            f"sibling dirs. Use Modly's model-download step (or the extension's "
            f"setup.py) to fetch them first."
        )

    def _find_weights_root(self, *subfolders):
        """Find a local dir that contains ALL given subfolders.

        Modly's model-download step may dump every node's weights into a single
        shared folder (often the extension's first node dir) rather than keeping
        them under each node's own model_dir. So we search self.model_dir AND
        its sibling node dirs (same parent) and return the first match.
        Returns a Path or None."""
        candidates = [self.model_dir]
        parent = self.model_dir.parent
        if parent.exists():
            candidates += [p for p in sorted(parent.iterdir()) if p.is_dir() and p != self.model_dir]
        for c in candidates:
            if all((c / sf).exists() for sf in subfolders):
                return c
        return None

    def _ensure_paint_weights(self, paint_subfolder: str = "hunyuan3d-paint-v2-0"):
        """Ensure the STANDARD (non-turbo) paint model + delight weights exist
        in Modly's local model dir (or a sibling node dir where Modly actually
        placed them). The turbo variant is ignored on purpose — its 2p5D UNet
        drops the reference-attention scale, so input views are not used.

        Returns the ROOT path (so from_pretrained can resolve delight +
        multiview as siblings of the subfolder) if usable, else None."""
        root = self._find_weights_root(paint_subfolder, "hunyuan3d-delight-v2-0")
        if root is not None:
            return str(root)
        # Not found locally. Surface a clear error instead of re-fetching.
        raise RuntimeError(
            f"Paint weights ({paint_subfolder} + hunyuan3d-delight-v2-0) not found "
            f"under {self.model_dir} or its sibling dirs. Use Modly's model-download "
            f"step (or the extension's setup.py) to fetch them first."
        )

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        # Route by node_id first (the host sends it, e.g. "generate"/"texture").
        # Fall back to input type: the texture node receives a mesh either as GLB
        # bytes or via the mesh_path param; everything else is mesh generation.
        node_id = str(params.get("node_id", "")).lower()
        # Detect texture_apply WITHOUT relying on node_id (Modly may not pass it
        # for extension nodes). The texture_apply node has an empty params_schema
        # while the texture node has specific params — use that as discriminator.
        _texture_node_keys = {"decimate_faces", "texture_size", "delight", "texture_diffusion_steps", "input_mode", "image_folder"}
        _has_texture_params = any(k in params for k in _texture_node_keys)
        _apply_bridge_exists = os.path.exists(str(EXT_DIR / "hunyuan3d_texture_apply_bridge.py"))
        _is_mesh_input = (
            image_bytes is not None
            and len(image_bytes) >= 4
            and image_bytes[:4] == b"glTF"
        )
        if _apply_bridge_exists and not _has_texture_params and ("mesh_path" in params or _is_mesh_input):
            return self.apply_texture_simple(image_bytes, params, progress_cb, cancel_event)
        _is_texture_request = (
            node_id == "texture"
            or _is_mesh_input
            or "mesh_path" in params
        )
        if _is_texture_request:
            return self.generate_texture(image_bytes, params, progress_cb, cancel_event)

        if self._model is None:
            self.load()

        num_inference_steps = int(params.get("num_inference_steps", 10))
        octree_resolution = int(params.get("octree_resolution", 512))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        dual_guidance_scale = float(params.get("dual_guidance_scale", 10.5))
        try:
            shape_views = int(params.get("shape_views", 4))
        except (TypeError, ValueError):
            shape_views = 4

        self._report(progress_cb, 5, "Preparing mesh generation…")
        self._check_cancelled(cancel_event)

        run = _modly_current_run_folder(self.outputs_dir, params)
        view_dir = run / "views_split"
        view_dir.mkdir(parents=True, exist_ok=True)
        output_path = run / "mesh.glb"

        input_mode = str(params.get("input_mode", "tiled")).lower()
        folder_path = params.get("image_folder", "")

        if input_mode == "folder" and folder_path:
            _auto_tile_folder(folder_path, run, shape_views)
            tiled_path = str(run / "folder_tiled.png")
            bridge_input_mode = "tiled"
            _cleanup_tmp = False
        else:
            import tempfile
            tmp_fd, tiled_path = tempfile.mkstemp(suffix=".png", prefix="hunyuan_in_")
            os.close(tmp_fd)
            img = Image.open(io.BytesIO(image_bytes))
            img.save(tiled_path)
            bridge_input_mode = input_mode
            _cleanup_tmp = True

            # Always remove backgrounds before the bridge sees the image
            if bridge_input_mode == "single":
                _rm_tmp = tiled_path.replace(".png", "_rm.png")
                if _rembg_single(tiled_path, _rm_tmp):
                    img = Image.open(_rm_tmp).convert("RGBA")
                    img.save(tiled_path)
                    try:
                        os.remove(_rm_tmp)
                    except Exception:
                        pass
            else:
                _split = Image.open(tiled_path).convert("RGBA")
                _w, _h = _split.size
                # Detect grid using the same closest-distance logic as
                # the bridge's _detect_grid so both agree on the layout.
                _img_aspect = _w / _h if _h > 0 else 1.0
                _candidates = [(2, 2), (2, 3), (3, 2)]
                _best, _best_err = (2, 2), float("inf")
                for _nc, _nr in _candidates:
                    _err = abs(_img_aspect - _nc / _nr)
                    if _err < _best_err:
                        _best, _best_err = (_nc, _nr), _err
                _ncols, _nrows = _best
                _cw, _ch = _w // _ncols, _h // _nrows
                _cell_positions = [(_cw * (i % _ncols), _ch * (i // _ncols)) for i in range(_ncols * _nrows)]
                _cells = []
                for _ci, (_cx, _cy) in enumerate(_cell_positions[:shape_views]):
                    _c = _split.crop((_cx, _cy, _cx + _cw, _cy + _ch))
                    _c_in = tiled_path.replace(".png", f"_c{_ci}.png")
                    _c_out = tiled_path.replace(".png", f"_c{_ci}_rm.png")
                    _c.save(_c_in)
                    if _rembg_single(_c_in, _c_out):
                        _c = Image.open(_c_out).convert("RGBA")
                        for _p in [_c_in, _c_out]:
                            try:
                                os.remove(_p)
                            except Exception:
                                pass
                    _cells.append(_c.resize((_cw, _ch)))
                _tile = Image.new("RGBA", (_w, _h), (0, 0, 0, 0))
                for _ci, _c in enumerate(_cells):
                    _cx = (_ci % _ncols) * _cw
                    _cy = (_ci // _ncols) * _ch
                    _tile.paste(_c, (_cx, _cy), _c)
                _tile.save(tiled_path)

        if bridge_input_mode == "single":
            shape_views = 1

        bridge = Path(__file__).resolve().parent / "hunyuan3d_bridge.py"

        _shape_model_dir = self._ensure_shape_weights()

        args = {
            "tiled_path": tiled_path,
            "input_mode": bridge_input_mode,
            "view_dir": str(view_dir),
            "output_path": str(output_path),
            "hy3dgen_path": str(self.hy3dgen_path),
            "model_cache": str(self.model_dir),
            "model_path": _shape_model_dir,
            "subfolder": "hunyuan3d-dit-v2-mv-turbo",
            "num_inference_steps": num_inference_steps,
            "octree_resolution": octree_resolution,
            "guidance_scale": guidance_scale,
            "dual_guidance_scale": dual_guidance_scale,
            "shape_views": shape_views,
        }

        env = os.environ.copy()
        env["HF_HUB_CACHE"] = str(self.model_dir)
        env["HY3DGEN_MODELS"] = str(self.model_dir)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        # Force the bridge's stdout/stderr unbuffered so warnings/tracebacks
        # and any percent/number progress stream live to the host console.
        env["PYTHONUNBUFFERED"] = "1"
        # Same allocator tuning as Era3D: keep committed memory out of slow
        # Windows shared GPU memory during the denoise/marching-cubes loop.
        # NOTE: expandable_segments is unsupported on this PyTorch/Windows
        # build (emits a warning and is ignored), so only max_split_size_mb.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # Bridge subprocess handles all progress from 10-100%.
        # Do not set a fixed percentage here — it creates a backward jump
        # (e.g. 20% → 10%) that the host may ignore or throttle.

        _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

        def _host_log(text: str) -> None:
            # Surface bridge stderr in the host console — the same place
            # Juggernaut's in-process print() warnings appear.
            text = _ANSI_RE.sub("", text).replace("\r", "").strip()
            if text:
                print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)

        def _pump_progress(pipe, progress_cb, sink):
            for raw in pipe:
                line = raw.strip()
                if line:
                    sink.append(line)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Not JSON. Skip anything that merely looks like a broken
                    # JSON fragment (starts with { or [); forward all other
                    # process output (warnings, errors, %/number lines) live.
                    if line and line[0] in "{[":
                        continue
                    _host_log(line)
                    continue
                t = msg.get("type")
                if t == "progress":
                    if progress_cb:
                        _step = msg.get("step", "")
                        _sub = msg.get("subtext", "")
                        if _sub:
                            _step = f"{_step} — {_sub}" if _step else _sub
                        progress_cb(msg.get("pct", 0), _step)
                elif t == "log":
                    _host_log(msg.get("message", ""))
                # done/error are surfaced by the generator on exit; ignore here.

        captured = []
        proc = subprocess.Popen(
            [str(self.python_path), str(bridge), json.dumps(args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        reader = threading.Thread(
            target=_pump_progress,
            args=(proc.stdout, progress_cb, captured),
            daemon=True,
        )
        reader.start()

        try:
            proc.wait(timeout=10800)  # 3h — 6GB card is slow; texture bake is heavy
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(
                "Mesh generation timed out after 3 hours. Likely OOM or a "
                "hung native call. Try octree_resolution=256 and disable texture."
            )

        reader.join()

        tail = "\n".join(captured)[-16000:]
        if proc.returncode != 0:
            raise RuntimeError(f"Mesh generation failed:\n{tail}")

        if not output_path.exists():
            raise RuntimeError(f"Output mesh not created:\n{tail}")

        # Clean up the temp input image (only if outside run dir)
        if _cleanup_tmp:
            try:
                os.remove(tiled_path)
            except OSError:
                pass

        # Remove the intermediate split-views folder (re-derived from the
        # tiled image); the run folder keeps only source/views/mesh.
        try:
            import shutil as _shutil
            _shutil.rmtree(view_dir, ignore_errors=True)
        except Exception:
            pass

        # Render orthographic views from the generated mesh
        self._report(progress_cb, 95, "Rendering views from mesh…")
        try:
            views_dir, depths_dir = _render_mesh_views(output_path, run)
            print(f"[{self.MODEL_ID}] Rendered views to {views_dir}", file=sys.stderr, flush=True)
        except Exception as e:
            # Non-fatal: mesh was generated successfully, views are extra
            print(f"[{self.MODEL_ID}] View rendering failed (non-fatal): {e}", file=sys.stderr, flush=True)

        self._report(progress_cb, 100, "Done")
        self.unload()
        return output_path

    def generate_texture(
        self,
        mesh_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Texture an existing mesh with the Hunyuan3D-2.0 paint pipeline.

        The mesh comes from the upstream Generate 3D Mesh node (run folder
        mesh.glb, or mesh_bytes / mesh_path). The conditioning image is a
        (tiled) image wired into the node — phase 1 uses its front (top-left)
        tile; later phases feed all four views. A manual image_path overrides
        the wired image.
        """
        if self._model is None:
            self.load()

        decimate_faces = int(params.get("decimate_faces", 40000))
        texture_size = int(params.get("texture_size", 2048))
        paint_model = params.get("paint_model", "tencent/Hunyuan3D-2")
        # ALWAYS use the STANDARD (non-turbo) paint model. The turbo variant's
        # 2p5D UNet skips the reference-attention `ref_scale_timing` assignment,
        # so the input views are ignored and every output view collapses to one
        # colour. The UI still offers a Turbo option, but we force standard here
        # because turbo produces broken (single-colour) textures AND its weights
        # are not downloaded (skipped in the manifest). Ignore the UI selection.
        paint_subfolder = "hunyuan3d-paint-v2-0"
        _snap_root = self._ensure_paint_weights(paint_subfolder)
        if _snap_root:
            # Pass the snapshot ROOT so from_pretrained resolves delight +
            # multiview as siblings of the subfolder.
            paint_model = str(_snap_root)

        self._report(progress_cb, 5, "Preparing texture…")
        self._check_cancelled(cancel_event)

        # Log inputs for debugging path issues
        _has_glb = mesh_bytes is not None and len(mesh_bytes) >= 4 and mesh_bytes[:4] == b"glTF"
        _raw_mesh_path = params.get("mesh_path", "")
        print(f"[tex] mesh_bytes GLB={_has_glb}, mesh_path='{_raw_mesh_path}', run_folder='{params.get('run_folder','')}'", file=sys.stderr, flush=True)

        run = _modly_current_run_folder(self.outputs_dir, params)
        output_path = run / "textured_mesh.glb"

        # Resolve the input mesh. A mesh wired into the node (mesh_bytes, GLB)
        # always wins over any stale mesh.glb in the run folder — otherwise a
        # previous shape output would shadow the user's mesh. Fall back to
        # run/mesh.glb only when nothing was wired in.
        #
        # The mesh-input node passes mesh_path as a path that may be relative to
        # the workspace (CWD is Modly's api dir), so resolve it against the
        # workspace root and the run folder before giving up.
        _wired_mesh = mesh_bytes is not None and mesh_bytes[:4] == b"glTF"
        mesh_path = params.get("mesh_path") or ""
        if _wired_mesh:
            import tempfile
            tmp_fd, mesh_path = tempfile.mkstemp(suffix=".glb", prefix="hunyuan_in_mesh_")
            os.close(tmp_fd)
            with open(mesh_path, "wb") as _f:
                _f.write(mesh_bytes)
        elif not mesh_path:
            mesh_path = str(run / "mesh.glb")

        def _resolve_mesh(p: str) -> str:
            if os.path.isabs(p) and os.path.exists(p):
                return p
            candidates = [
                p,
                os.path.join(str(self.outputs_dir.parent), p) if hasattr(self, "outputs_dir") else p,
                os.path.abspath(p),
                os.path.join(str(run), os.path.basename(p)),
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    return c
            # Log the candidates we tried so the user can see what went wrong.
            tried = [c for c in candidates if c]
            print(f"[mesh] Could not resolve mesh path '{p}'. Tried: {tried}", file=sys.stderr, flush=True)
            return p

        mesh_path = _resolve_mesh(mesh_path)
        if not os.path.exists(mesh_path):
            raise RuntimeError(f"Texture node: input mesh not found at {mesh_path}")

        # Resolve the conditioning image based on input_mode
        input_mode = str(params.get("input_mode", "tiled")).lower()
        folder_path = params.get("image_folder", "")
        _cleanup_img = False

        if input_mode == "folder" and folder_path:
            _ref_count = int(params.get("reference_images", 4) or 4)
            _auto_tile_folder(folder_path, run, shape_views=_ref_count)
            image_path = str(run / "folder_tiled.png")
            bridge_input_mode = "tiled"
        else:
            image_path = params.get("image_path", "")
            if not image_path and mesh_bytes and mesh_bytes[:4] != b"glTF":
                import tempfile
                _fd, _img_path = tempfile.mkstemp(suffix=".png", prefix="hunyuan_tex_in_")
                os.close(_fd)
                with open(_img_path, "wb") as _f:
                    _f.write(mesh_bytes)
                image_path = _img_path
                _cleanup_img = True
            bridge_input_mode = input_mode

        if bridge_input_mode == "single":
            reference_images = 1

        bridge = Path(__file__).resolve().parent / "hunyuan3d_texture_bridge.py"

        args = {
            "mesh_path": mesh_path,
            "output_path": str(output_path),
            "hy3dgen_path": str(self.hy3dgen_path),
            "model_cache": str(self.model_dir),
            "decimate_faces": decimate_faces,
            "texture_size": texture_size,
            "image_path": image_path,
            "input_mode": bridge_input_mode,
            "paint_model": paint_model,
            "paint_subfolder": paint_subfolder,
            "reference_images": int(params.get("reference_images", 4) or 4),
            "delight": str(params.get("delight", "off") or "off").lower(),
            "texture_diffusion_steps": int(params.get("texture_diffusion_steps", 30) or 30),
            "texture_method": str(params.get("texture_method", "diffusion") or "diffusion").lower(),
        }

        env = os.environ.copy()
        env["HF_HUB_CACHE"] = str(self.model_dir)
        env["HY3DGEN_MODELS"] = str(self.model_dir)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        self._report(progress_cb, 20, "Painting texture (Hunyuan3D-2.0)…")

        _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

        def _host_log(text: str) -> None:
            text = _ANSI_RE.sub("", text).replace("\r", "").strip()
            if text:
                print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)

        def _pump_progress(pipe, progress_cb, sink):
            for raw in pipe:
                line = raw.strip()
                if line:
                    sink.append(line)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    if line and line[0] in "{[":
                        continue
                    _host_log(line)
                    continue
                t = msg.get("type")
                if t == "progress":
                    if progress_cb:
                        _step = msg.get("step", "")
                        _sub = msg.get("subtext", "")
                        if _sub:
                            _step = f"{_step} — {_sub}" if _step else _sub
                        progress_cb(msg.get("pct", 0), _step)
                elif t == "log":
                    _host_log(msg.get("message", ""))

        captured = []
        proc = subprocess.Popen(
            [str(self.python_path), str(bridge), json.dumps(args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        reader = threading.Thread(
            target=_pump_progress,
            args=(proc.stdout, progress_cb, captured),
            daemon=True,
        )
        reader.start()

        try:
            proc.wait(timeout=10800)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("Texture bake timed out after 3 hours.")

        reader.join()
        tail = "\n".join(captured)[-16000:]
        if proc.returncode != 0:
            raise RuntimeError(f"Texture bake failed:\n{tail}")
        if not output_path.exists():
            raise RuntimeError(f"Textured mesh not created:\n{tail}")

        try:
            if mesh_path.endswith(".glb") and "hunyuan_in_mesh_" in mesh_path:
                os.remove(mesh_path)
        except OSError:
            pass
        if _cleanup_img:
            try:
                os.remove(image_path)
            except OSError:
                pass

        # Render orthographic views from the textured mesh (same preview the
        # Generate node produces for the 3D scene display).
        self._report(progress_cb, 95, "Rendering views from mesh…")
        try:
            views_dir, _depths_dir = _render_mesh_views(output_path, run)
            print(f"[{self.MODEL_ID}] Rendered views to {views_dir}", file=sys.stderr, flush=True)
        except Exception as e:
            # Non-fatal: mesh was textured successfully, views are extra
            print(f"[{self.MODEL_ID}] View rendering failed (non-fatal): {e}", file=sys.stderr, flush=True)

        self._report(progress_cb, 100, "Done")
        self.unload()
        return output_path

    def apply_texture_simple(self, image_bytes, params, progress_cb=None, cancel_event=None):
        """texture_apply node: map texture atlas image(s) onto an already UV'd mesh.

        No GPU, no diffusion, no bake — just trimesh + PIL (+ pygltflib for PBR).
        The mesh comes from the upstream node (GLB bytes or mesh_path), the
        albedo from image_path / wired image bytes / texture_folder. Optional
        normal, roughness/specular and displacement maps (explicit paths or
        texture_folder scan) are packed into standard glTF PBR slots.
        Output: PBR GLB in the run folder.
        """
        self._report(progress_cb, 5, "Preparing texture apply…")
        self._check_cancelled(cancel_event)

        run = _modly_current_run_folder(self.outputs_dir, params)
        # Distinct from the Texture Mesh node's textured_mesh.glb so chaining
        # the nodes can't overwrite the texture node's output.
        output_path = run / "pbr_mesh.glb"

        _is_glb = image_bytes is not None and len(image_bytes) >= 4 and image_bytes[:4] == b"glTF"
        _is_img = image_bytes is not None and not _is_glb

        # Resolve input mesh
        import tempfile
        mesh_path = params.get("mesh_path") or ""
        if _is_glb:
            tmp_fd, mesh_path = tempfile.mkstemp(suffix=".glb", prefix="hunyuan_in_mesh_")
            os.close(tmp_fd)
            with open(mesh_path, "wb") as _f:
                _f.write(image_bytes)
        elif not mesh_path:
            mesh_path = str(run / "mesh.glb")

        def _resolve_mesh(p):
            if os.path.isabs(p) and os.path.exists(p):
                return p
            candidates = [
                p,
                os.path.join(str(self.outputs_dir.parent), p) if hasattr(self, "outputs_dir") and self.outputs_dir else p,
                os.path.abspath(p),
                os.path.join(str(run), os.path.basename(p)),
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    return c
            tried = [c for c in candidates if c]
            print(f"[texture_apply] Could not resolve mesh path '{p}'. Tried: {tried}", file=sys.stderr, flush=True)
            return p

        mesh_path = _resolve_mesh(mesh_path)
        if not os.path.exists(mesh_path):
            raise RuntimeError(f"texture_apply: input mesh not found at {mesh_path}")

        # Resolve texture image (albedo). A texture_folder with a detectable
        # albedo file is an alternative to wiring an image. Albedo Source
        # = folder ignores the wired image entirely and uses the folder's
        # texturemap (the image input can't be unwired on this node).
        _cleanup_img = False
        _albedo_source = str(params.get("albedo_source", "image") or "image").lower()
        _use_wired = (_albedo_source != "folder")
        texture_path = params.get("image_path", "") if _use_wired else ""
        texture_folder = params.get("texture_folder", "") or ""
        if not texture_path and _is_img and _use_wired:
            _fd, texture_path = tempfile.mkstemp(suffix=".png", prefix="hunyuan_tex_in_")
            os.close(_fd)
            with open(texture_path, "wb") as _f:
                _f.write(image_bytes)
            _cleanup_img = True
        if not texture_path and not texture_folder:
            raise RuntimeError("texture_apply: no texture image provided (wire an image, set image_path, or set texture_folder)")

        def _num(key, default=0.0):
            try:
                return float(params.get(key, default) or 0)
            except (TypeError, ValueError):
                return float(default)

        def _onoff(key, default="off"):
            return str(params.get(key, default) or default).lower() == "on"

        args = {
            "mesh_path": mesh_path,
            "texture_path": texture_path,
            "mode": str(params.get("mode", "pbr") or "pbr").lower(),
            "texture_folder": texture_folder,
            "normal_scale": _num("normal_scale", 1.0) or 1.0,
            "normal_flip_y": "on" if _onoff("normal_flip_y", "off") else "off",
            "smooth_normals": "on" if _onoff("smooth_normals", "on") else "off",
            "output_path": str(output_path),
        }
        bridge = str(EXT_DIR / "hunyuan3d_texture_apply_bridge.py")
        if not os.path.exists(bridge):
            raise RuntimeError(f"Bridge not found: {bridge}")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        captured = []

        def _host_log(text: str) -> None:
            text = (text or "").strip()
            if text:
                print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)

        def _pump(stream):
            try:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    captured.append(line)
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    t = msg.get("type")
                    if t == "progress":
                        if progress_cb:
                            _step = msg.get("step", "")
                            _sub = msg.get("subtext", "")
                            if _sub:
                                _step = f"{_step} — {_sub}" if _step else _sub
                            progress_cb(msg.get("pct", 0), _step)
                    elif t == "log":
                        _host_log(msg.get("message", ""))
            except Exception as e:
                captured.append(f"[pump-error] {e!r}")

        proc = subprocess.Popen(
            [str(self.python_path), str(bridge), json.dumps(args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        reader.start()

        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("texture_apply timed out after 10 minutes.")

        reader.join()
        tail = "\n".join(captured)[-16000:]
        if proc.returncode != 0:
            raise RuntimeError(f"texture_apply failed:\n{tail}")

        # The bridge may rename pbr_mesh.glb -> partial_pbr_mesh.glb (or vice
        # versa) depending on what PBR maps were found. Check both names.
        if not output_path.exists():
            _alt = "partial_pbr_mesh.glb" if output_path.name == "pbr_mesh.glb" else "pbr_mesh.glb"
            _alt_path = run / _alt
            if _alt_path.exists():
                output_path = _alt_path
            else:
                raise RuntimeError(f"Textured mesh not created:\n{tail}")

        try:
            if mesh_path.endswith(".glb") and "hunyuan_in_mesh_" in mesh_path:
                os.remove(mesh_path)
        except OSError:
            pass
        if _cleanup_img:
            try:
                os.remove(texture_path)
            except OSError:
                pass

        # Render orthographic views from the PBR mesh (same preview the
        # Generate node produces for the 3D scene display).
        self._report(progress_cb, 95, "Rendering views from mesh…")
        try:
            views_dir, _depths_dir = _render_mesh_views(output_path, run)
            print(f"[{self.MODEL_ID}] Rendered views to {views_dir}", file=sys.stderr, flush=True)
        except Exception as e:
            # Non-fatal: mesh was textured successfully, views are extra
            print(f"[{self.MODEL_ID}] View rendering failed (non-fatal): {e}", file=sys.stderr, flush=True)

        self._report(progress_cb, 100, "Done")
        return output_path


    @classmethod
    def params_schema(cls) -> list:
        # Mirror of the manifest "generate" node params (manifest is authoritative).
        return [
            {
                "id": "num_inference_steps",
                "label": "Quality Steps",
                "type": "select",
                "default": 10,
                "options": [
                    {"value": 5, "label": "5 (Turbo - Fast)"},
                    {"value": 10, "label": "10 (Turbo - Balanced)"},
                    {"value": 30, "label": "30 (Standard - High)"},
                ],
                "tooltip": "Turbo variant uses 5-10 steps. Standard uses 30.",
            },
            {
                "id": "octree_resolution",
                "label": "Mesh Resolution",
                "type": "select",
                "default": 256,
                "options": [
                    {"value": 32, "label": "32 (Blocky - fewest faces)"},
                    {"value": 64, "label": "64 (Low - 6GB)"},
                    {"value": 96, "label": "96 (Low)"},
                    {"value": 128, "label": "128 (Balanced - 6GB)"},
                    {"value": 192, "label": "192 (Fast - 6GB)"},
                    {"value": 256, "label": "256 (Low)"},
                    {"value": 380, "label": "380 (Standard)"},
                    {"value": 512, "label": "512 (High)"}
                ],
                "tooltip": "Marching-cubes grid resolution. Lower = coarser mesh, fewer faces, less VRAM.",
            },
            {
                "id": "dual_guidance_scale",
                "label": "Dual Guidance",
                "type": "select",
                "default": "10.5",
                "options": [
                    {"value": "10.5", "label": "On (Best quality, 3x slower)"},
                    {"value": "-1", "label": "Off (2x faster, slight quality loss)"}
                ],
                "tooltip": "Dual-guidance CFG triples compute per step on 6GB. Turn off for speed.",
            },
            {
                "id": "shape_views",
                "label": "Input Views",
                "type": "select",
                "default": 4,
                "options": [
                    {"value": 1, "label": "1 view (front)"},
                    {"value": 2, "label": "2 views (front + left)"},
                    {"value": 3, "label": "3 views (front + left + back)"},
                    {"value": 4, "label": "4 views (all)"},
                ],
                "tooltip": "How many cardinal views to use for shape generation (1-6).",
            },
            {
                "id": "input_mode",
                "label": "Image Input Mode",
                "type": "select",
                "default": "tiled",
                "options": [
                    {"value": "single", "label": "Single Image (whole image = main view)"},
                    {"value": "tiled", "label": "Tiled Image (auto-detect grid from image)"},
                    {"value": "folder", "label": "Image Folder (auto-tile from folder)"},
                ],
                "tooltip": "How to interpret the wired image. 'Single' uses the whole image as one view. 'Tiled' splits a pre-made grid (auto-detects 2x2/2x3/3x2 from aspect ratio). 'Folder' reads images from the folder path below.",
            },
            {
                "id": "image_folder",
                "label": "Image Folder",
                "type": "string",
                "default": "",
                "tooltip": "Only used when Input Mode = 'Image Folder'. Path to a folder of images to auto-tile into a 2x2 grid.",
            },
        ]
