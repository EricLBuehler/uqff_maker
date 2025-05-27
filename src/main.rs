use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use mistralrs::{IsqType, TextModelBuilder, VisionModelBuilder};

// Names follow the format: [name][version]-[size]-[instruct?]-[quant].uqff

#[cfg(feature = "cuda")]
const QUANTIZATIONS: &[IsqType] = &[
    IsqType::Q2K,
    IsqType::Q3K,
    IsqType::Q4K,
    IsqType::Q5K,
    IsqType::Q8_0,
    IsqType::F8E4M3,
];

#[cfg(not(feature = "cuda"))]
const QUANTIZATIONS: &[IsqType] = &[
    IsqType::AFQ2,
    IsqType::AFQ3,
    IsqType::AFQ4,
    IsqType::AFQ6,
    IsqType::AFQ8,
];

fn default_template(model_name: &str) -> String {
    // Lowercase and remove the first hyphen to merge name and version
    let mut s = model_name.to_lowercase();
    if let Some(pos) = s.find('-') {
        s.remove(pos);
    }
    // Append the placeholder and extension
    format!("{s}-###.uqff")
}

#[derive(Parser, Debug)]
/// The model generated is output to a directory with the same name as the model ID.
/// For example, the model ID `username/model_name` will generate the quantizations to `model_name`.
struct Args {
    /// Model ID to generate the UQFF model for.
    #[arg(short, long)]
    model_id: String,

    /// Template filename. If not specified, will be derived from the model ID.
    #[arg(short, long)]
    filename: Option<String>,

    /// To quantize a vision model, add this flag.
    #[arg(short, long, value_enum)]
    vision: bool,

    /// Directory to save the generated files. Defaults to model name.
    #[arg(short = 'd', long)]
    save_dir: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model_id = &args.model_id;
    let model_name = Path::new(model_id)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_id);

    // Use save_dir as parent directory if provided, else current directory; model_name is always the subdirectory
    let output_dir = if let Some(ref dir) = args.save_dir {
        PathBuf::from(dir).join(model_name)
    } else {
        PathBuf::from(model_name)
    };
    std::fs::create_dir_all(&output_dir)?;

    // Determine the template: use provided filename or derive from model_name
    let template: String = if let Some(f) = &args.filename {
        f.clone()
    } else {
        default_template(model_name)
    };

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
