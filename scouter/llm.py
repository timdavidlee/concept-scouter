from openai import OpenAI


def load_openai_key():
    file = "/Users/timlee/Dropbox/keys/openai_key.txt"
    with open(file, "r") as f:
        return f.read()


def get_openai_client():
    api_key = load_openai_key()
    client = OpenAI(api_key=api_key)
    return client
