# `uqff_maker`

Automated generation of UQFF models with mistral.rs.

## Example usage
```
cargo run --features ... -- -m mistralai/Devstral-Small-2505

python3 upload.py -- --model-id mistralai/Devstral-Small-2505 --token ... --username ...
```
## Example usage for a vision model
```
cargo run --features ... -- -m google/gemma-3-4b-it --vision

python3 upload.py -- --model-id google/gemma-3-4b-it --token ... --username ...
```

Then, use [`generate_uqff_card.py`](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/generate_uqff_card.py) to create the model card.
