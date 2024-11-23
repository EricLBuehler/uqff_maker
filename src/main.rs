#![allow(dead_code, unused_imports)]

use anyhow::Result;
use mistralrs::{IsqType, TextModelBuilder, VisionLoaderType, VisionModelBuilder};

// Names follow the format: [name][version]-[size]-[instruct?]-[quant].uqff

const TEXT_MODELS_TO_QUANTIZE: &[(&str, &str)] = &[
    // // Gemma
    // ("google/gemma-1.1-2b-it", "gemma1.1-2b-instruct-###.uqff"),
    // ("google/gemma-1.1-7b-it", "gemma1.1-7b-instruct-###.uqff"),
    // // Gemma 2
    // ("google/gemma-2-2b-it", "gemma2-2b-instruct-###.uqff"),
    // ("google/gemma-2-9b-it", "gemma2-9b-instruct-###.uqff"),
    // ("google/gemma-2-27b-it", "gemma2-9b-instruct-###.uqff"),
    // // Llama
    // (
    //     "meta-llama/Llama-3.2-1B-Instruct",
    //     "llama3.2-1b-instruct-###.uqff",
    // ),
    // (
    //     "meta-llama/Llama-3.2-3B-Instruct",
    //     "llama3.2-3b-instruct-###.uqff",
    // ),
    // (
    //     "meta-llama/Llama-3.1-8B-Instruct",
    //     "llama3.1-8b-instruct-###.uqff",
    // ),
    // // Mistral
    // (
    //     "mistralai/Mistral-7B-Instruct-v0.3",
    //     "mistral0.3-7b-instruct-###.uqff",
    // ),
    // (
    //     "mistralai/Mistral-Nemo-Instruct-2407",
    //     "mistral-nemo-2407-instruct-###.uqff",
    // ),
    // (
    //     "mistralai/Mistral-Small-Instruct-2409",
    //     "mistral-small-2409-instruct-###.uqff",
    // ),
    // Mixtral
    // (
    //     "mistralai/Mixtral-8x7B-Instruct-v0.1",
    //     "mixtral0.1-8x7b-instruct-###.uqff",
    // ),
    // // Phi 3
    // (
    //     "microsoft/Phi-3.5-mini-instruct",
    //     "phi3.5-mini-instruct-###.uqff",
    // ),
    (
        "microsoft/Phi-3.5-MoE-instruct",
        "phi-moe3.5-instruct-###.uqff",
    ),
];

const VISION_MODELS_TO_QUANTIZE: &[(&str, &str, VisionLoaderType)] = &[
    // Phi 3.5 Vision
    (
        "microsoft/Phi-3.5-vision-instruct",
        "phi3.5-vision-instruct-###.uqff",
        VisionLoaderType::Phi3V,
    ),
    // Llama 3.2 Vision
    (
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llam3.2-vision-instruct-###.uqff",
        VisionLoaderType::VLlama,
    ),
];

const QUANTIZATIONS: &[IsqType] = &[
    IsqType::Q3K,
    IsqType::Q4K,
    IsqType::Q5K,
    IsqType::Q8_0,
    IsqType::HQQ4,
    IsqType::HQQ8,
    IsqType::F8E4M3,
];

#[tokio::main]
async fn main() -> Result<()> {
    for (model, template) in TEXT_MODELS_TO_QUANTIZE {
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

            let res = TextModelBuilder::new(model)
                .with_isq(*quant)
                .write_uqff(uqff_file.into())
                .with_logging()
                .build()
                .await;

            match res {
                Ok(_) => (),
                Err(e) => {
                    println!("{}  Error! {e:?}\n", "\n".repeat(3));
                }
            }
        }
    }

    // for (model, template, tp) in VISION_MODELS_TO_QUANTIZE {
    //     println!("{}\n{model} \n{}", "=".repeat(20), "=".repeat(20));

    //     for quant in QUANTIZATIONS {
    //         let dir = model.split('/').last().unwrap();
    //         let uqff_file = format!(
    //             "{dir}/{}",
    //             template.replace("###", &format!("{quant:?}").to_lowercase())
    //         );

    //         std::fs::create_dir_all(&dir)?;

    //         println!(
    //             "{}  Generating with quantization {quant:?} to {uqff_file}{}",
    //             "\n".repeat(3),
    //             "\n".repeat(3)
    //         );

    //         let res = VisionModelBuilder::new(model, tp.clone())
    //             .with_isq(*quant)
    //             .write_uqff(uqff_file.into())
    //             .with_logging()
    //             .build()
    //             .await;

    //         match res {
    //             Ok(_) => (),
    //             Err(e) => {
    //                 println!("{}  Error! {e:?}\n", "\n".repeat(3));
    //             }
    //         }
    //     }
    // }

    Ok(())
}
