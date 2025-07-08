# `uqff_maker`

Automated generation of UQFF models with mistral.rs.

## Example usage
```
# Generate UQFF
cargo run --features ... -- quantize -m mistralai/Devstral-Small-2505

# Make the model card
cargo run --features ... -- model-card -w Devstral-Small-2505

python3 upload.py --model_id mistralai/Devstral-Small-2505 --token ... --username ...
```

## Example usage for a vision model
```
# Generate UQFF
cargo run --features ... -- quantize -m google/gemma-3-4b-it --vision

# Make the model card
cargo run --features ... -- model-card -w gemma-3-4b-it

python3 upload.py --model_id google/gemma-3-4b-it --token ... --username ...
```
