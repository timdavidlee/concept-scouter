# creates the training data spacy binary file
python ner_cli.py llm2spacy

# trains the actual model
python ner_cli.py train_model