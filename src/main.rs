use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use mistralrs::{IsqType, TextModelBuilder, VisionLoaderType, VisionModelBuilder};

// Names follow the format: [name][version]-[size]-[instruct?]-[quant].uqff

const QUANTIZATIONS: &[IsqType] = &[
    IsqType::Q2K,
    IsqType::Q3K,
    IsqType::Q4K,
    IsqType::Q5K,
    IsqType::Q8_0,
    IsqType::AFQ2,
    IsqType::AFQ3,
    IsqType::AFQ4,
    IsqType::AFQ6,
    IsqType::AFQ8,
    IsqType::F8E4M3,
];

#[derive(Parser, Debug)]
/// The model generated is output to a directory with the same name as the model ID.
/// For example, the model ID `username/model_name` will generate the quantizations to `model_name`.
struct Args {
    /// Model ID to generate the UQFF model for.
    #[arg(short, long)]
    model_id: String,

    /// Template filename. The `###` placeholder will be replaced by each quantization.
    #[arg(short, long)]
    filename: String,

    /// To quantize a vision model, add this flag.
    #[arg(short, long, value_enum)]
    vision: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model_id = &args.model_id;
    let template = &args.filename;
    let model_name = Path::new(model_id)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_id);
    let output_dir = PathBuf::from(model_name);
    std::fs::create_dir_all(&output_dir)?;

    println!("{}\n{model_id} \n{}", "=".repeat(20), "=".repeat(20));

    for quant in QUANTIZATIONS {
        let filename = template.replace("###", &format!("{quant:?}").to_lowercase());
        let uqff_path = output_dir.join(&filename);

        println!(
            "{}  Generating with quantization {quant:?} to {}{}",
            "\n".repeat(3),
            uqff_path.display(),
            "\n".repeat(3)
        );

        let result = if args.vision {
            VisionModelBuilder::new(model_id)
                .with_isq(*quant)
                .write_uqff(uqff_path.clone().into())
                .with_logging()
                .build()
                .await
        } else {
            TextModelBuilder::new(model_id)
                .with_isq(*quant)
                .write_uqff(uqff_path.clone().into())
                .with_logging()
                .build()
                .await
        };
        if let Err(e) = result {
            eprintln!("Error generating {}: {:?}", uqff_path.display(), e);
        }
    }

    Ok(())
}
