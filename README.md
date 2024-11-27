# `uqff_maker`

Automated generation of UQFF models with mistral.rs.

## Example usage
```
cargo run --features cuda -release -- --model-id meta-llama/Llama-3.2-3B-Instruct --filename llama3.2-3b-instruct-###.uqff

python3 upload.py -- --model-id meta-llama/Llama-3.2-3B-Instruct --token ... --username ...
```
## Example usage for a vision model
```
cargo run --features metal -release -- --model-id Qwen/Qwen2-VL-2B-Instruct --filename qwen2vl-2b-instruct-###.uqff --vision-arch qwen2vl

python3 upload.py -- --model-id Qwen/Qwen2-VL-2B-Instruct --token ... --username ...
```
