import sys
from huggingface_hub import HfApi, create_repo

repo_id = "Ypeng12/InferRoute"
print(f"Checking/Creating Space repository '{repo_id}' on Hugging Face...")

api = HfApi()

try:
    url = create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
    print(f"Space repository ready at: {url}")
except Exception as e:
    print(f"Repo notice: {e}")

print("Uploading InferRoute + Quant.ai files to Hugging Face Space...")

api.upload_folder(
    folder_path="c:/Users/pengy/OneDrive/Desktop/InferRoute",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        "*.pyc",
        "__pycache__/*",
        "**/__pycache__/*",
        ".git/*",
        "**/.git/*",
        ".venv/*",
        "**/.venv/*",
        "node_modules/*",
        "**/node_modules/*",
        "scratch/*",
        "*.pdf",
        ".gemini/*",
        "inferroute.db"
    ]
)

print("Upload completed successfully!")
print(f"Space live URL: https://huggingface.co/spaces/{repo_id}")
