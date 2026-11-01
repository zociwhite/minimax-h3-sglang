"""RunPod serverless worker: MiniMax H3 lewat SGLang (tanpa ComfyUI).

Sekali per worker: unduh checkpoint (scoped ke satu varian), nyalakan
`sglang serve`, tunggu siap. Tiap job cukup proxy HTTP ke server lokal itu,
jadi model tetap hangat selama worker hidup (idleTimeout 2 menit).

Container disk ephemeral: worker mati -> model hilang -> unduh ulang.
Itu konsekuensi "tanpa network volume".
"""
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

MODEL_ID = os.environ.get("MODEL_ID", "MiniMaxAI/MiniMax-H3")
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "fl2va").lower()
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30010"))
BASE = f"http://127.0.0.1:{SGLANG_PORT}"
BOOT_TIMEOUT_S = int(os.environ.get("SGLANG_BOOT_TIMEOUT_S", "2400"))
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "900"))

_proc = None
_local_model = None


def _variant_dir():
    return MODEL_VARIANT.upper()


def ensure_model():
    """Unduh hanya yang dipakai satu varian (~134 GB), bukan seluruh repo (464 GB)."""
    global _local_model
    if _local_model:
        return _local_model
    from huggingface_hub import snapshot_download

    pattern = f"{_variant_dir()}/*"
    print(f"[model] snapshot_download {MODEL_ID} include={pattern}", flush=True)
    start = time.time()
    _local_model = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["model_index.json", pattern],
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    print(f"[model] siap di {_local_model} ({time.time() - start:.0f}s)", flush=True)
    return _local_model


def serve_cmd(model_path):
    return [
        "sglang", "serve",
        "--model-path", model_path,
        "--model-variant", MODEL_VARIANT,
        "--quantization", os.environ.get("SGLANG_QUANT", "convrot_int8"),
        "--attention-backend", os.environ.get("SGLANG_ATTENTION", "fa"),
        "--performance-mode", "memory",
        "--layerwise-offload-components", "dit,text_encoder",
        "--dit-offload-prefetch-size", "1",
        "--dit-layerwise-resident-layers", os.environ.get("DIT_RESIDENT_LAYERS", "4"),
        "--enable-torch-compile", "false",
        "--host", "127.0.0.1",
        "--port", str(SGLANG_PORT),
    ]


def _http(method, path, payload=None, timeout=120, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return body if raw else json.loads(body)


def _ready():
    try:
        _http("GET", "/health", timeout=5)
        return True
    except urllib.error.HTTPError as exc:
        # HTTP server hidup walau rute /health tidak ada di build ini.
        return exc.code in (404, 405)
    except Exception:
        return False


def boot():
    global _proc
    if _proc is not None:
        return
    cmd = serve_cmd(ensure_model())
    print("[boot] " + " ".join(cmd), flush=True)
    _proc = subprocess.Popen(cmd)
    start = time.time()
    while time.time() - start < BOOT_TIMEOUT_S:
        if _proc.poll() is not None:
            raise RuntimeError(f"sglang mati dini (exit {_proc.returncode})")
        if _ready():
            print(f"[boot] sglang siap dalam {time.time() - start:.0f}s", flush=True)
            return
        time.sleep(5)
    raise RuntimeError("sglang tidak siap sebelum boot timeout")


def _cond_from(data, role, frame_index=None):
    """Tulis base64/URL jadi file lokal, kembalikan kondisi SGLang."""
    if isinstance(data, dict):
        data = data.get("data") or data.get("url") or data.get("image") or ""
    if not isinstance(data, str) or not data:
        raise ValueError("kondisi gambar kosong")
    if data.startswith("http://") or data.startswith("https://"):
        uri = data
    else:
        raw = data.split(",", 1)[1] if data.startswith("data:") else data
        path = f"/tmp/cond_{role}_{int(time.time() * 1000)}.png"
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(raw))
        uri = f"file://{path}"
    cond = {"type": "image", "uri": uri, "role": role}
    if frame_index is not None:
        cond["frame_index"] = frame_index
    return cond


def build_payload(inp):
    task = (inp.get("task") or "t2va").lower()
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("'prompt' wajib diisi")

    conditions = []
    images = list(inp.get("images") or [])
    if inp.get("image"):
        images.insert(0, inp["image"])
    for idx, item in enumerate(images[:2]):
        role = "keyframe" if task == "fl2va" else "reference"
        frame_index = idx if task == "fl2va" else None
        conditions.append(_cond_from(item, role, frame_index))

    seconds = float(inp.get("seconds") or 5)
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "seconds": int(seconds),
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": int(inp.get("short_edge") or 768),
            "aspect_ratio": inp.get("aspect_ratio") or "16:9",
            "duration_seconds": seconds,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(inp.get("steps") or 50),
        "seed": int(inp.get("seed") or 1101),
    }
    if inp.get("quality"):
        payload["quality"] = inp["quality"]
    if inp.get("flow_shift"):
        payload["flow_shift"] = float(inp["flow_shift"])
    return payload


def _wait_video(video_id):
    deadline = time.time() + JOB_TIMEOUT_S
    while time.time() < deadline:
        state = _http("GET", f"/v1/videos/{video_id}", timeout=60)
        status = str(state.get("status") or "").lower()
        if status in ("completed", "succeeded", "success", "done"):
            return state
        if status in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"render gagal: {state}")
        time.sleep(5)
    raise TimeoutError(f"video {video_id} tidak selesai dalam {JOB_TIMEOUT_S}s")


def handler(job):
    inp = job.get("input") or {}
    if inp.get("action") == "health":
        return {"ok": True, "variant": MODEL_VARIANT, "model": MODEL_ID,
                "booted": _proc is not None, "ready": _ready()}

    boot()
    payload = build_payload(inp)
    created = _http("POST", "/v1/videos", payload, timeout=300)
    video_id = created.get("id")
    if not video_id:
        raise RuntimeError(f"tidak dapat id video: {created}")
    print(f"[job] submitted {video_id}", flush=True)

    _wait_video(video_id)
    blob = _http("GET", f"/v1/videos/{video_id}/content?variant=0", timeout=300, raw=True)
    return {
        "video_b64": base64.b64encode(blob).decode(),
        "bytes": len(blob),
        "format": "mp4",
        "variant": MODEL_VARIANT,
        "task": payload["task"],
        "prompt": payload["prompt"],
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
