"""python -m scouter.llm_to_span_data"""
import json
from pathlib import Path

import spacy
from spacy.util import filter_spans
from spacy.tokens import Span, Doc, DocBin
from spacy.matcher import PhraseMatcher

DISK_LOCATION = "/tmp/train.spacy"
LLM_GENERATED_DIR = Path("./sample_data/llm_generated/")
SPACY_DOC_DIR = Path("./sample_data/spacy_docs/")


def load_json(fl: Path):
    with open(fl, "r") as f:
        data = json.load(f)
        return data


def multi_word_to_single_pattern(label: str, input_text):
    return dict(
        label=label,
        pattern=[{"LOWER": tok.lower()} for tok in input_text.split(" ")]
    )


def create_spacy_phrase_matcher(nlp, terms):
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(text.lower()) for text in terms]
    matcher.add("TerminologyList", patterns)
    return matcher


def extract_spans(doc, matcher, label):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label) for _, start, end in matches]
    return spans


def convert_generated_text_to_annotated_payload(data: dict):
    nlp = spacy.blank("en")

    company_matcher = create_spacy_phrase_matcher(nlp, [data["company"],])
    company_category_matcher = create_spacy_phrase_matcher(nlp, [data["company_category"],])

    brand_category = set([row[0].lower() for row in data["brands"]])
    brand_category_matcher = create_spacy_phrase_matcher(nlp, brand_category)

    brands = set([row[1].lower() for row in data["brands"]])
    brand_matcher = create_spacy_phrase_matcher(nlp, brands)

    doc = nlp(data["response_text"].lower())
    company_spans = extract_spans(doc, company_matcher, "COMPANY")
    company_cat_spans = extract_spans(doc, company_category_matcher, "COMPANY_CATEGORY")
    brand_spans = extract_spans(doc, brand_matcher, "BRAND")
    brand_cat_spans = extract_spans(doc, brand_category_matcher, "BRAND_CATEGORY")

    all_spans = company_cat_spans + company_spans + brand_spans + brand_cat_spans

    # because a category could be `Computer Accessories` and also `Accessories`, generally take
    # the longer one first
    all_spans = filter_spans(all_spans)
    output_doc = nlp(data["response_text"].lower())
    output_doc.set_ents(all_spans)

    return output_doc


def parse_llm_text_for_spacy(force: bool = False, train_file: str = DISK_LOCATION):
    jsonfiles = list(LLM_GENERATED_DIR.glob("*"))
    docbin = DocBin()
    for fl in jsonfiles:
        savefile = SPACY_DOC_DIR / fl.name

        if savefile.exists() and (not force):
            print("already converted, loading directly from disk: {}".format(savefile))
            spacy_doc = load_json_formatted_spacy_doc(savefile)
            docbin.add(spacy_doc)
            continue

        data = load_json(fl)
        spacy_doc = convert_generated_text_to_annotated_payload(data)
        docbin.add(spacy_doc)

        with open(savefile, "w") as f:
            f.write(json.dumps(spacy_doc.to_json(), indent=2))
            print("saved: {}".format(savefile))

    ndocs_collected = len(docbin)
    docbin.to_disk(train_file)
    print("spacy-serialized version saved to {:,}: {}".format(ndocs_collected, train_file))


def load_json_formatted_spacy_doc(file: str) -> Doc:
    nlp = spacy.blank("en")
    with open(file, "r") as f:
        doc = Doc(nlp.vocab).from_json(json.load(f))
        return doc


def color_ent_doc(doc: Doc):
    colors = {
        "COMPANY": "#B99095",
        "COMPANY_CATEGORY": "#FCB5AC",
        "BRAND": "#B5E5CF",
        "BRAND_CATEGORY": "#3D5B59"
    }
    options = {"colors": colors}
    spacy.displacy.render(doc, style="ent", options=options, jupyter=True)