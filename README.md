# `uqff_maker`

Automated generation of UQFF models with mistral.rs.

## Example usage
```
cargo run --features cuda -release -- --model-id meta-llama/Llama-3.2-3B-Instruct --filename llama3.2-3b-instruct-###.uqff

python3 upload.py -- --model-id meta-llama/Llama-3.2-3B-Instruct --token ... --username ...
```
