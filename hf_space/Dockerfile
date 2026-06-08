FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo
RUN git clone https://github.com/edarsem/chessformer.git .

# Python deps + huggingface_hub for checkpoint download
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

# Download checkpoint from HF model hub at build time.
# Set HF_REPO to your model repo ID (e.g. "edarsem/chessformer").
ARG HF_REPO=edarsem/chessformer
ARG CKPT_FILE=chessformer_v0.pt
RUN python - <<'EOF'
import os, sys
from huggingface_hub import hf_hub_download
repo  = os.environ.get("HF_REPO", "edarsem/chessformer")
fname = os.environ.get("CKPT_FILE", "chessformer_v0.pt")
os.makedirs("checkpoints", exist_ok=True)
path = hf_hub_download(repo_id=repo, filename=fname, local_dir="checkpoints")
print(f"Checkpoint ready: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
EOF

EXPOSE 7860

# open_browser=false: don't try to open a browser in the container
# wandb.enabled=false: no W&B in the Space
# hydra.run.dir: write Hydra outputs to /tmp (writable in HF Spaces)
CMD ["python", "scripts/serve.py", \
     "checkpoint=checkpoints/chessformer_v0.pt", \
     "port=7860", \
     "open_browser=false", \
     "wandb.enabled=false", \
     "hydra.run.dir=/tmp/hydra"]
