# concept-scouter
ner -> LLM -> ner


```
python cli.py --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

Commands:
  format-kaggle  Formats `company` and `brand` data from kaggle
  gen-llm-text   Generates both prompts + retrieves LLM responses for...
  llm2spacy      Converts the synthetic, llm-
  train_model    Trains a custom NER model
```

## Dataset

```
https://www.kaggle.com/datasets/charanpuvvala/company-classification
https://www.kaggle.com/datasets/carrie1/ecommerce-data
https://www.kaggle.com/datasets/hma2022/amazon-global-store-us-from-saudi-souq
```


## installing spacy

```sh
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
```

## How to get a `base_config.cfg` for spacy

https://spacy.io/usage/training

What to do with the base config

```sh
python -m spacy init fill-config scouter/base_config.cfg config.cfg
python -m spacy init fill-config scouter/tfmr_config.cfg config.cfg
```


## How to train

```
python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy --gpu-id 0
```

but for our purposes:

```
python -m spacy train config.cfg --output ./spacy-output/ --paths.train /tmp/train.spacy --paths.dev /tmp/train.spacy

python -m spacy train config.cfg --output ./spacy-output/ --paths.train /tmp/train.spacy --paths.dev /tmp/train.spacy --gpu-id 0
```