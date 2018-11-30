import spacy

nlp = spacy.load("en")
from spacy.lang.en.stop_words import STOP_WORDS

import pandas as pd

from data import t0, t1, t2, t3, t4, t5, t6

def lemmatize(doc):
    doc = nlp(doc)
    lemma_list = [token.lemma_ for token in doc
                  if not token.is_punct
                  and not token.is_space
                  and (token.text == "US" or not token.lower_ in STOP_WORDS)
                  and not token.tag == "POS"]
    return (lemma_list)


t_list = [t0, t1, t2, t3, t4, t5, t6]


def jaccard(doc1, doc2):
    set1 = set(doc1)
    set2 = set(doc2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

lems = [lemmatize(t) for t in t_list]


def lemmatize(doc):
    return [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_space
           and (token.text == "US" or not token.lower_ in STOP_WORDS)
           and not token.tag_ == "POS"
    ]


def tf(s, doc):
    lemma_list = lemmatize(doc)
    s = " ".join(lemmatize(nlp(s)))
    return lemma_list.count(s)


text_list = [t0, t1, t2, t3, t4, t5, t6]
doc_list = [nlp(i) for i in text_list]

def idf(s, doc_list):
    counter = 0
    for doc in doc_list:
        if tf(s, doc) > 0:
            counter += 1
    if counter > 0:
        return 1 / counter
    else:
        return 0


def tf_idf(s, doc, doc_list):
    return tf(s, doc) * idf(s, doc_list)


def all_lemmas(doc_list):
    lems = set()
    for doc in doc_list:
        lems.update(lemmatize(doc))
    return lems


def tf_idf_doc(doc, doc_list):
    lemmas = all_lemmas(doc_list)
    lemma_dict = {lemma: tf_idf(lemma, doc, doc_list) for lemma in lemmas}
    return lemma_dict


def tf_idf_scores(doc_list):
    df = pd.DataFrame([tf_idf_doc(doc, doc_list) for doc in doc_list])
    return df
