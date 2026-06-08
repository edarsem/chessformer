FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone latest code
RUN git clone https://github.com/edarsem/chessformer.git .

# CPU-only torch (saves ~1.5 GB vs the CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# App dependencies (no torchaudio / ipykernel / wandb)
RUN pip install --no-cache-dir \
    chess>=1.10.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    polars>=0.20.0 \
    pyarrow>=15.0.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    python-multipart>=0.0.9 \
    rich>=13.0.0 \
    huggingface_hub>=0.23.0

# Download checkpoint from the model repo at build time
ARG HF_REPO=edarsem/chessformer
ARG CKPT_FILE=chessformer_v0.pt
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('checkpoints', exist_ok=True); \
p = hf_hub_download(repo_id='${HF_REPO}', filename='${CKPT_FILE}', local_dir='checkpoints'); \
print('Checkpoint:', p, f'({os.path.getsize(p)/1e6:.1f} MB)') \
"

EXPOSE 7860

CMD ["python", "scripts/serve.py", \
     "checkpoint=checkpoints/chessformer_v0.pt", \
     "port=7860", \
     "open_browser=false", \
     "wandb.enabled=false", \
     "hydra.run.dir=/tmp/hydra"]
