# pip install huggingface_hub

from huggingface_hub import HfApi
import huggingface_hub

MODELS = [
    # # Gemma
    # ("google/gemma-1.1-2b-it", "gemma1.1-2b-instruct-###.uqff"),
    # ("google/gemma-1.1-7b-it", "gemma1.1-7b-instruct-###.uqff"),
    # # Gemma 2
    # ("google/gemma-2-2b-it", "gemma2-2b-instruct-###.uqff"),
    # ("google/gemma-2-9b-it", "gemma2-9b-instruct-###.uqff"),
    ("google/gemma-2-27b-it", "gemma2-27b-instruct-###.uqff"),
    # Llama
    ("meta-llama/Llama-3.2-1B-Instruct", "llama3.2-1b-instruct-###.uqff"),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama3.2-3b-instruct-###.uqff"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b-instruct-###.uqff"),
    # Mistral
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral0.3-7b-instruct-###.uqff"),
    ("mistralai/Mistral-Nemo-Instruct-2407", "mistral-nemo-2407-instruct-###.uqff"),
    ("mistralai/Mistral-Small-Instruct-2409", "mistral-small-2409-instruct-###.uqff"),
    # Mixtral
    # ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral0.1-8x7b-instruct-###.uqff"),
    # Phi 3
    ("microsoft/Phi-3.5-mini-instruct", "phi3.5-mini-instruct-###.uqff"),
    # ("microsoft/Phi-3.5-MoE-instruct", "phi-moe3.5-instruct-###.uqff"),
    ("microsoft/Phi-3.5-vision-instruct", "phi3.5-vision-instruct-###.uqff"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", "llam3.2-vision-instruct-###.uqff"),
]

TOKEN = "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"

api = HfApi(token=TOKEN)

for model, template in MODELS:
    src_dir = model.split("/")[-1]
    tgt_repo_id = f"EricB/{src_dir}-UQFF"

    print(f"{'='*20}\n{model} to {tgt_repo_id}\n{'='*20}")

    api.create_repo(
        repo_id=tgt_repo_id,
        private=True,
        exist_ok=True
    )

    huggingface_hub.upload_folder(
        repo_id=tgt_repo_id,
        folder_path=src_dir,
        commit_message="Upload model",
        token=TOKEN
    )
