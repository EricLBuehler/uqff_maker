# pip install huggingface_hub

from huggingface_hub import HfApi
import huggingface_hub
import argparse

parser = argparse.ArgumentParser(description="Process command-line arguments.")

# Add arguments
parser.add_argument(
    "-m",
    "--model_id",
    required=True,
    help="The original Hugging Face model ID. The model name itself is assumed to be a local directory containing the UQFF quantization.",
)
parser.add_argument("-t", "--token", required=True, help="Hugging Face write token.")
parser.add_argument("-u", "--username", required=True, help="Hugging Face username to upload the model to.")

# Parse the arguments
args = parser.parse_args()

api = HfApi(token=args.token)

src_dir = args.model_id.split("/")[-1]
tgt_repo_id = f"{args.username}/{src_dir}-UQFF"

print(f"{'='*20}\n{args.model_id} to {tgt_repo_id}\n{'='*20}")

api.create_repo(repo_id=tgt_repo_id, private=True, exist_ok=True)

huggingface_hub.upload_folder(
    repo_id=tgt_repo_id,
    folder_path=src_dir,
    commit_message="Upload model",
    token=args.token,
)
