import os
from huggingface_hub import snapshot_download

#os.environ["HF_HUB_DISABLE_XET"] = "1"

OBSERVER = "tiiuae/Falcon3-1B-Base"
PERFORMER = "tiiuae/Falcon3-1B-Instruct"

for model_id in [OBSERVER, PERFORMER]:
    print(f"Downloading {model_id} ...")
    snapshot_download(repo_id=model_id)
    print(f"Done: {model_id}")

print("All models downloaded.")
