use std::str::FromStr;

use anyhow::Result;
use clap::Parser;
use mistralrs::{IsqType, TextModelBuilder, VisionLoaderType, VisionModelBuilder};

// Names follow the format: [name][version]-[size]-[instruct?]-[quant].uqff

const QUANTIZATIONS: &[IsqType] = &[
    IsqType::Q3K,
    IsqType::Q4K,
    IsqType::Q5K,
    IsqType::Q8_0,
    IsqType::HQQ4,
    IsqType::HQQ8,
    IsqType::F8E4M3,
];

#[derive(Parser)]
/// The model generated is output to a directory with the same name as the model ID.
/// For example, the model ID `username/model_name` will generate the quantizations to `model_name`.
struct Args {
    /// Model ID to generate the UQFF model for.
    #[arg(short, long)]
    model_id: String,

    /// Template filename, roughly with the format: `[name][version]-[size]-[instruct?]-###.uqff`.
    /// The ### will be automatically replaced with the quantization and must be specified.
    #[arg(short, long)]
    filename: String,

    /// To quantize a vision model, you must specify this flag. See the mistral.rs docs:
    /// https://ericlbuehler.github.io/mistral.rs/mistralrs/enum.VisionLoaderType.html
    #[arg(short, long)]
    vision_arch: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model = &args.model_id;
    let template = &args.filename;

    println!("{}\n{model} \n{}", "=".repeat(20), "=".repeat(20));

    for quant in QUANTIZATIONS {
        let dir = model.split('/').last().unwrap();
        let uqff_file = format!(
            "{dir}/{}",
            template.replace("###", &format!("{quant:?}").to_lowercase())
        );

        std::fs::create_dir_all(&dir)?;

        println!(
            "{}  Generating with quantization {quant:?} to {uqff_file}{}",
            "\n".repeat(3),
            "\n".repeat(3)
        );

        let res = if let Some(vision_arch) = &args.vision_arch {
            let ty = VisionLoaderType::from_str(vision_arch).map_err(anyhow::Error::msg)?;
            VisionModelBuilder::new(model, ty)
                .with_isq(*quant)
                .write_uqff(uqff_file.into())
                .with_logging()
                .build()
                .await
        } else {
            TextModelBuilder::new(model)
                .with_isq(*quant)
                .write_uqff(uqff_file.into())
                .with_logging()
                .build()
                .await
        };

        match res {
            Ok(_) => (),
            Err(e) => {
                println!("{}  Error! {e:?}\n", "\n".repeat(3));
            }
        }
    }

    Ok(())
}
