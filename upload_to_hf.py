from huggingface_hub import HfApi, login
import os

# Login - this will open browser for authentication
login()

# Upload entire folder to your Space
api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="nabilyasini/StereoAwareGNN1",
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".git", "*.pth", "upload_to_hf.py"]
)

print("Upload complete!")
