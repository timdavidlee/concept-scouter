"""python -m cli"""
import os
import typer
from pathlib import Path

from spacy.cli.init_config import fill_config
from spacy.cli.train import train

from sample_data.kaggle import format_kaggle_data
from scouter.prompts import gen_llm_prompts_and_responses
from scouter.llm_to_span_data import parse_llm_text_for_spacy


app = typer.Typer()

@app.command("format-kaggle")
def cli_format_kaggle_data():
    """
    Formats `company` and `brand` data from kaggle
    """
    format_kaggle_data()


@app.command("gen-llm-text")
def cli_gen_llm_text(n_companies: int = 25, brands_per: int = 5):
    """
    Generates both prompts + retrieves LLM responses for
        n_companies = number of companies
        brands_per = the number of brands that company interacts with
    """
    gen_llm_prompts_and_responses(
        n_prompts=n_companies,
        brands_per=brands_per
    )

@app.command("llm2spacy")
def cli_llm2spacy(force: bool = False):
    """
    Converts the synthetic, llm-generated text into spacy compatible json
    """
    parse_llm_text_for_spacy(force=force)


@app.command("train_model")
def cli_init_config(transformer_flag: bool = False):
    base_path = "./scouter/tfmr_config.cfg" if transformer_flag else "./scouter/base_config.cfg"
    fill_config(
        output_file=Path("config.cfg"),
        base_path=Path(base_path)
    )

    output_path = "spacy-output/tfmr/" if transformer_flag else "spacy-output/base/"
    os.makedirs(output_path, exist_ok=True)
    overrides = {
        "paths.train": "/tmp/train.spacy",
        "paths.dev": "/tmp/train.spacy",
    }

    train(
        config_path=Path("config.cfg"),
        output_path=Path(output_path),
        overrides=overrides
    )


if __name__ == "__main__":
    app()
