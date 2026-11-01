# MiniMax H3 — SGLang Worker

Repo serverless RunPod untuk MiniMax H3 menggunakan backend resmi SGLang (tanpa ComfyUI).

- **Base image**: `lmsysorg/sglang:latest`
- **Output image**: `z0ci/minimax-h3-sglang:latest` (via GitHub Actions)
- **Model**: `MiniMaxAI/MiniMax-H3` (scoped `FL2VA/*`)
- **Backend**: SGLang native async HTTP API (`POST /v1/videos`)
- **Hardware**: RunPod pool `ADA_48_PRO` (RTX PRO 6000 MIG 48GB / L40S, $1.75/hr)
- **Container disk**: 160 GB (ephemeral, unduh on-demand)
- **Idle timeout**: 120 detik (2 menit)

## Secrets GitHub yang dibutuhkan
1. `DOCKERHUB_USERNAME`: akun Docker Hub (contoh: `z0ci`)
2. `DOCKERHUB_TOKEN`: Personal Access Token Docker Hub (Read/Write)
