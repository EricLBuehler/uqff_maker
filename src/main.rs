use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use dialoguer::{Confirm, Input};
use mistralrs::{IsqType, TextModelBuilder, VisionModelBuilder};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Write;

// Names follow the format: [name][version]-[size]-[instruct?]-[quant].uqff

const QUANTIZATIONS: &[IsqType] = &[
    IsqType::Q2K,
    IsqType::Q3K,
    IsqType::Q4K,
    IsqType::Q5K,
    IsqType::Q8_0,
    IsqType::F8E4M3,
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
enum Commands {
    /// Generate UQFF models.
    Quantize(Args),
    /// Generate a Hugging Face model card for a UQFF model.
    ModelCard(ModelCardArgs),
}

#[derive(Parser, Debug)]
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

#[derive(Parser, Debug)]
/// Arguments for the `modelcard` sub‑command.
struct ModelCardArgs {
    /// Directory containing UQFF files. If omitted, you will be prompted.
    #[arg(short = 'w', long)]
    work_dir: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Commands::parse();

    match cli {
        Commands::Quantize(args) => {
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
        Commands::ModelCard(card_args) => {
            // Optional working directory provided on the CLI
            let work_dir_opt = card_args.work_dir.clone();
            // Model card generation logic (ported from Python)
            let msg = "This script is used to generate a Hugging Face model card.";
            println!("{}", "-".repeat(msg.len()));
            println!("{}", msg);
            println!("{}", "-".repeat(msg.len()));

            let model_id: String = Input::new()
                .with_prompt("Please enter the original model ID")
                .interact_text()?;
            let display_model_id: String = Input::new()
                .with_prompt("Please enter the model ID where this model card will be displayed")
                .interact_text()?;
            let is_vision: bool = Confirm::new()
                .with_prompt("Is this a vision model?")
                .interact()?;

            let mut output = format!(
                r#"---
tags:
  - uqff
  - mistral.rs
base_model: {model_id}
base_model_relation: quantized
---

<!-- Autogenerated from user input. -->

"#,
            );

            output += &format!("# `{model_id}`, UQFF quantization\n\n");

            output += r#"
Run with [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Documentation: [UQFF docs](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/UQFF.md).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
3) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.
"#;

            println!(" NOTE: Getting metadata now, press CTRL-C when you have entered all files");
            println!(" NOTE: If multiple quantizations were used: enter the quantization names, and then in the next prompt, the topology file used.");

            output += "\n## Examples\n";
            output += "|Quantization type(s)|Example|\n|--|--|\n";

            let mut topologies: HashMap<String, String> = HashMap::new();
            let mut n = 0;

            // Determine directory containing UQFF files
            let uqff_dir_path = if let Some(ref dir) = work_dir_opt {
                PathBuf::from(dir)
            } else {
                let uqff_dir: String = Input::new()
                    .with_prompt("Enter the directory containing UQFF files (leave empty for current directory)")
                    .allow_empty(true)
                    .interact_text()?;
                if uqff_dir.trim().is_empty() {
                    std::env::current_dir()?
                } else {
                    PathBuf::from(uqff_dir.trim())
                }
            };

            // Collect and group `.uqff` files (group by prefix, ignoring trailing numeric suffix)
            let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
            for entry in std::fs::read_dir(&uqff_dir_path)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let path = entry.path();
                    if path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map_or(false, |e| e.eq_ignore_ascii_case("uqff"))
                    {
                        // Derive grouping key: drop trailing numeric index if present
                        let stem = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or_default();
                        let key = if let Some((pre, suf)) = stem.rsplit_once('-') {
                            if suf.chars().all(|c| c.is_ascii_digit()) {
                                pre.to_string()
                            } else {
                                stem.to_string()
                            }
                        } else {
                            stem.to_string()
                        };
                        groups.entry(key).or_default().push(path);
                    }
                }
            }

            // Iterate through each group (sorted by key)
            for paths in groups.values() {
                // Use the first file in the group as the representative example
                let path = &paths[0];
                let file = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                println!("Processing group: {}", file);

                // Guess quantization name (handle optional numeric suffix)
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                let parts: Vec<&str> = stem.split('-').collect();
                let default_quant = if parts.len() >= 2
                    && parts.last().unwrap().chars().all(|c| c.is_ascii_digit())
                {
                    parts[parts.len() - 2].to_uppercase()
                } else {
                    parts.last().unwrap_or(&"").to_uppercase()
                };

                // Ask user for confirmation or override
                let quants_input: String = Input::new()
                    .with_prompt(format!(
                        "Enter quantization NAMES used for files with prefix \"{}\" (default: {default_quant})",
                        stem
                    ))
                    .allow_empty(true)
                    .interact_text()?;

                let quants = if quants_input.trim().is_empty() {
                    default_quant.clone()
                } else {
                    quants_input.to_uppercase()
                };

                if quants.contains(',') {
                    let quants_vec: Vec<String> =
                        quants.split(',').map(|x| x.trim().to_uppercase()).collect();
                    let topology: String = Input::new()
                        .with_prompt("Enter topology used to make UQFF with multiple quantizations")
                        .interact_text()?;
                    topologies.insert(file.clone(), topology.clone());
                    output += &format!("|{} (see topology for this file)|", quants_vec.join(","));
                } else {
                    output += &format!("|{}|", quants.trim());
                }

                let cmd = if is_vision { "vision-plain" } else { "plain" };
                output +=
                    &format!("`./mistralrs-server -i {cmd} -m {display_model_id} -f {file}`|\n");

                n += 1;
            }

            if n == 0 {
                eprintln!("Need at least one file");
                return Ok(());
            }

            if !topologies.is_empty() {
                output += "\n\n## Topologies\n**The following model topologies were used to generate this UQFF file. Only information pertaining to ISQ is relevant.**\n";
                for (name, file) in topologies {
                    let contents = std::fs::read_to_string(&file)
                        .unwrap_or_else(|_| "<Could not read topology file>".to_string());
                    output += &format!("### Used for `{}`\n\n", name);
                    output += &format!("```yml\n{}\n```\n", contents);
                }
            }

            let msg = "Done! Please enter the output filename";
            println!("\n{}", "-".repeat(msg.len()));
            println!("{}", msg);
            println!("{}", "-".repeat(msg.len()));

            let out: String = Input::new()
                .with_prompt("Enter the output filename")
                .interact_text()?;
            let out_path = if Path::new(&out).is_absolute() {
                PathBuf::from(&out)
            } else {
                uqff_dir_path.join(&out)
            };
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&out_path)?;
            file.write_all(output.as_bytes())?;
            println!("Model card written to {}", out_path.display());
            Ok(())
        }
    }
}
