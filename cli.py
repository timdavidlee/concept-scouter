"""python -m cli"""
import typer

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


if __name__ == "__main__":
    app()
