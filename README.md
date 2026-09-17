# Hunyuan3D-2mv — Modly Extension (v2.2)

Generate textured 3D meshes from images using Tencent's Hunyuan3D-2mv. Three nodes:

| Node | What it does |
|------|-------------|
| **Generate 3D Mesh** | Creates a 3D shape (untextured GLB mesh) from 1–6 photos of an object |
| **Texture Mesh** | Paints a texture onto an existing mesh using reference photos |
| **Apply Texture** | Maps an existing UV atlas (e.g. externally super-resolved) onto a UV'd mesh — no diffusion |

### What's new in v2.2
- **Self-contained install** — `setup.py` now installs *every* runtime dependency (torch + hy3dgen + scipy + scikit-image + onnxruntime + rembg + trimesh + …) into the isolated venv, including a `pip install .` packaging mode for GitHub users. No manual dependency steps.
- **Machine-specific CUDA rasterizer** — the texture renderer is compiled during installation for the detected NVIDIA GPU instead of shipping a binary for one fixed GPU architecture. The generated `.pyd` is stored in the extension and copied into its venv.
- **Weights stay in Modly** — all model weights are still downloaded through the extension's nodes in Modly's **Extensions → model** view (per-node `hf_repo` / download check). Nothing model-related ships in the repo.
- **First-load bridge** — new `hunyuan3d_bootstrap.py` runs on first load after install/download and bridges anything living *outside* the extension dir:
  - weights Modly placed in any node model dir / HF hub cache are hardlinked into the layout the pipeline expects (instant, zero extra disk, fully offline afterwards),
  - the venv's machine-specific `custom_rasterizer` CUDA kernel is re-installed if missing/stale,
  - a state file (`.bridge_state.json`, gitignored) makes every later run a no-op.
- **Bridges stay authoritative** — both generation bridges (shape + texture) re-run the (idempotent) first-load bridge right before loading models, so any change since the last run is patched into out-of-extension files automatically on the first generate.

### What was new in v2.0
- **Automatic background removal** — rembg runs on every input image before processing. No toggle needed.
- **Image Folder mode** — point to a folder of photos; filenames (`front`, `left`, `back`, `right`) determine grid placement. Auto-tiles and saves `folder_tiled.png` for reuse.
- **Hybrid texture method** — bakes real reference pixels onto the views they cover via a smooth thin-plate-spline silhouette warp and uses diffusion for uncovered views. Best with orthographic/isometric references (e.g. the MV-Adapter grid).
- **Delight toggle** — turn off to skip the delight model (~1.5 GB VRAM saved) and keep original colors.
- **Progress tracking** — multiview diffusion step progress shown in the node status.
- **Debug views** — per-view split outputs saved to a `views_split` folder for troubleshooting.
- **Texture white-background normalization** — reference images for the texture multiview diffusion are always composited onto white.
- **Adjustable diffusion steps** — both mesh generation and texture multiview diffusion expose step counts as numeric parameters (1-60 for mesh, 5-60 for texture).

---

## CUDA Build Prerequisites

Texture generation uses a native CUDA rasterizer compiled for the installed GPU during extension setup. Install these **before** installing or reinstalling the extension:

- An up-to-date NVIDIA display driver
- The CUDA Toolkit matching the CUDA version selected by the extension's PyTorch wheel (normally CUDA 12.4 for pre-Blackwell GPUs and CUDA 12.8 for newer GPUs)
- Visual Studio Build Tools with the **Desktop development with C++** workload
- The Windows SDK included in that workload

The CUDA Toolkit is required to build the rasterizer, not to run it after the build. It may be uninstalled after setup succeeds, but it must be installed again before a future extension reinstall or rebuild. The NVIDIA display driver must remain installed.

If setup reports that `nvcc` or a C++ compiler is missing, install the prerequisites above and reinstall the extension. Setup detects the GPU capability, builds `custom_rasterizer_kernel`, and copies the resulting `.pyd` into the extension's virtual environment.

### GPU Architecture Support

The installer chooses the PyTorch CUDA wheel from the GPU's compute capability, not only from the installed driver version:

| Compute capability | Typical GPU families | Installer path |
|---|---|---|
| `sm_20`–`sm_37` | Fermi / Kepler | Unsupported |
| `sm_50`–`sm_62` | Maxwell / Pascal | PyTorch CUDA 12.4 |
| `sm_70`–`sm_90` | Volta / Turing / Ampere / Ada / Hopper | PyTorch CUDA 12.4 |
| `sm_100` / `sm_120` | Blackwell | PyTorch CUDA 12.8 |

Hunyuan3D also needs approximately 6 GB or more of VRAM in practice; a GPU may be CUDA-compatible but still run out of memory during generation.

## Updating / Installing from GitHub

This repo is the canonical source: `https://github.com/iammojogo-sudo/hunyuan3d-2mv-2.0.1_modly`

1. **Install the CUDA build prerequisites above.**
2. **Add the extension in Modly's Extensions tab** (paste the repo URL) — Modly downloads it and re-runs `setup.py`, which installs every dependency (torch, hy3dgen, rembg, …) and compiles the GPU-specific rasterizer into the isolated venv.
3. **Download the model weights** in the Extensions → model view (per-node **Download** button) — ~15 GB total from public Hugging Face repos.
4. **Model weights do not need re-downloading on updates** — they already live in Modly's `models/` folder; the first-load bridge links them into place automatically.

For a manual install:

```
pip install https://github.com/iammojogo-sudo/hunyuan3d-2mv-2.0.1_modly/archive/refs/heads/main.zip
```

The manual package install also requires the CUDA build prerequisites above when texture generation is needed.

## Background Removal (rembg)

All input images automatically have their background removed via rembg before being fed to the mesh or texture pipeline. This happens for every input mode:

- **Single / Tiled mode:** rembg runs on the wired image before splitting or forwarding.
- **Folder mode:** rembg runs on each individual file in the folder before padding/tiling. Intermediate files (`rembg_*.png`, `padded_*.png`) are saved in the current Modly run folder.

Background removal uses a dedicated Python venv and the ONNX Runtime runs on CPU to avoid CUDA conflicts with PyTorch.

---

## Image Input Modes

Both nodes accept images in three ways. Select via the **Image Input Mode** dropdown in the node settings:

### Single Image — `input_mode: single`
Wire in **one photo** of your subject. The whole image is used as a single front view. Best for when you only have one angle of the object.

### Tiled Image — `input_mode: tiled`
Wire in a pre-made grid image. A 2×2 grid uses front, left, back, right; a
2×3 or 3×2 grid can provide front, left, back, right, top, bottom. The bridge
auto-detects the layout from the image aspect ratio.

### Image Folder — `input_mode: folder`
Put your photos in a folder, paste the folder path into the **Image Folder** field (or use the folder picker). The extension auto-tiles up to 6 images into a 2×2 grid for up to 4 views or a 2×3 grid for 5-6 views, then saves the composite (`folder_tiled.png`) to the run folder for reuse.

**Folder naming guide:** The extension reads filenames to place them in the correct grid position:
| Filename contains | Grid position |
|---|---|
| `front` | Top-left |
| `left` | Top-right |
| `back` | Bottom-left |
| `right` | Bottom-right |

If filenames don't contain any of these keywords, they fill remaining slots alphabetically.

---

## Step-by-Step: Generate 3D Mesh

### Method A: Wire a single photo (easiest)
1. Drop an image node and wire it into **Generate 3D Mesh**
2. Set **Image Input Mode** → `Single Image`
3. Set **Input Views** → `1 view (front)`
4. Click Generate — background removed automatically

### Method B: Wire a tiled image
1. Create a 2×2, 2×3, or 3×2 grid image in reading order
2. Wire it into **Generate 3D Mesh**
3. Set **Image Input Mode** → `Tiled Image`
4. Set **Input Views** → how many views to use (1–6)
5. Click Generate

### Method C: Use a folder of images
1. Place 1–6 photos in a folder (name them with `front`, `left`, `back`, `right`, `top`, or `bottom` in the filename)
2. Set **Image Input Mode** → `Image Folder`
3. Paste the folder path into **Image Folder** or use the folder picker
4. Set **Input Views** → how many views to use (1–6)
5. Click Generate — the composite tile (`folder_tiled.png`) is saved in the run folder for later use

### Key Parameters

| Parameter | What it does |
|-----------|-------------|
| **Quality Steps** | Number of shape-generation diffusion steps (1-60). 5-10 = turbo fast, 30 = standard high quality |
| **Mesh Resolution** | Lower = coarser mesh, less VRAM. 128–256 for 6GB cards, 380+ for 12GB+ |
| **Dual Guidance** | On = best quality but 3× slower. Off = faster, slight quality loss |
| **Input Views** | How many of the 1-6 available views to feed the model |
| **Image Input Mode** | Single / Tiled / Folder — how to interpret the input |

---

## Step-by-Step: Texture Mesh

1. **Generate 3D Mesh** first, or wire in your own GLB mesh
2. Wire the mesh output into **Texture Mesh**
3. Wire reference image(s) into the second input
4. Set **Image Input Mode**:
   - `Single Image` — one reference photo
   - `Tiled Image` — a 2×2, 2×3, or 3×2 tile with up to 6 reference views
   - `Image Folder` — folder of reference photos (auto-tiled)
5. Set **Reference Images** to how many views to use for conditioning
6. Set **Texture Method**:
   - `Diffusion` — multiview model generates all 6 views (smoother on spheres/organic shapes)
   - `Hybrid` — bakes your real reference pixels onto the views they cover (thin-plate-spline silhouette warp at full resolution) and synthesizes the rest with diffusion. Sharper on hard-surface objects; best with orthographic/isometric references like the MV-Adapter grid. Single photos are fine too — only the front view uses the photo, the rest come from diffusion.
7. Set **Delight**:
   - `On` — runs the delight model to normalize lighting (can wash out colors, but may look more synthetic)
   - `Off` — skips the delight model, saves ~1.5 GB VRAM, keeps original image colors
8. Set **Texture Diffusion Steps** (5-60, default 30). Lower = faster/rougher; higher = slower/potentially sharper
9. Click Generate

### Key Parameters

| Parameter | What it does |
|-----------|-------------|
| **Texture Resolution** | Higher = sharper but more VRAM. 1024 is the sweet spot |
| **Decimate Faces** | Reduces mesh face count before UV unwrap. Lower = faster, less VRAM |
| **Reference Images** | 1–6 real views wired in (front, left, back, right, top, bottom). The front view conditions diffusion; Hybrid also uses the other supplied views directly in the texture bake |
| **Texture Method** | Diffusion = model generates every view from the front ref; Hybrid = real pixels on covered views + diffusion for the rest (best with orthographic MV-Adapter refs). The retired `deform` value is treated as `hybrid`. |
| **Delight** | Off = keep real colors (saves ~1.5 GB VRAM). On = normalize lighting |
| **Texture Diffusion Steps** | Number of multiview diffusion steps for texture generation (5-60). 5 = fast/rough, 30 = default, 60 = slow/sharp |
| **Image Input Mode** | Single / Tiled / Folder |

---

## Step-by-Step: Apply Texture

Use **Apply Texture** when the mesh already has UV coordinates and you have an
albedo texture or a complete PBR texture set. This node does not load diffusion
models and does not perform a bake.

1. Wire a UV'd mesh into **Apply Texture**.
2. For **Albedo Source = Wired image**, wire the albedo image into the image
   input.
3. For **Albedo Source = Folder**, set **Texture Folder** to the folder that
   contains the maps.
4. Select `Albedo` to apply only the albedo, or `PBR` to also load optional
   normal, roughness, and metallic maps.
5. Adjust normal scale, normal Y flipping, and smooth normals if needed.

When scanning a folder, the node recognizes `texturemap` for albedo,
`normalmap`, `roughnessmap`, and `metallicmap`. Missing optional maps are
skipped. A mesh without UVs must go through **Texture Mesh** or another UV
unwrap tool first.

---

## Requirements

- **VRAM:** 6 GB minimum, 8 GB+ recommended
- **GPU:** NVIDIA with CUDA
- **Disk:** ~15 GB free for model weights
- **Python:** 3.11 (installed automatically into the extension venv)
- **Dependencies:** all installed by `setup.py` (torch, hy3dgen, diffusers, transformers, rembg, scipy, scikit-image, onnxruntime, trimesh, …)

---

## Troubleshooting

### "No input image (tiled_path) found"
- Make sure you wired an image into the node
- If using **Image Folder** mode, verify the folder path is correct

### Texture bake fails / black mesh
- Wire at least one image to the Texture Mesh node
- Try **Reference Images** = 1 first, then increase
- Check that background was cleanly removed (rembg runs automatically)

### CUDA out of memory
- Lower **Mesh Resolution** to 128 or 64
- Use **Quality Steps** = 5 (Turbo)
- Turn off **Dual Guidance**
- Set **Texture Resolution** to 512 or lower
- Set **Delight** to Off (saves ~1.5 GB VRAM)

### "CUDA Toolkit was not found" / rasterizer build failure
Install the CUDA Toolkit, Visual Studio C++ Build Tools, and the Windows SDK before reinstalling the extension. The rasterizer is compiled during setup and cannot be repaired from a generic prebuilt binary after installation.

### "No CUDA GPUs are available"
Kill lingering Python processes in Task Manager and restart Modly. If persistent, your GPU driver may need a reboot.

### Images placed in wrong grid position
In **Image Folder** mode, name your files with `front`, `left`, `back`,
`right`, `top`, or `bottom` in the filename (for example,
`myobject_front.png`). The extension reads these keywords to place them
correctly.
