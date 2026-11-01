FROM lmsysorg/sglang:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=MiniMaxAI/MiniMax-H3 \
    MODEL_VARIANT=fl2va \
    SGLANG_PORT=30010 \
    JOB_TIMEOUT_S=900

RUN pip install "runpod>=1.12" "huggingface_hub[cli]" hf_transfer

WORKDIR /app
COPY handler.py /app/handler.py

# Model tidak dibakar ke image: diunduh ke container disk saat worker start,
# discope ke satu varian saja (FL2VA +/-134 GB). Butuh container disk >= 160 GB.
CMD ["python", "-u", "/app/handler.py"]
