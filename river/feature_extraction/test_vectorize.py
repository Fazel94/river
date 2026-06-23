from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river import feature_extraction


@pytest.mark.parametrize(
    "params, text, expected_ngrams",
    [
        pytest.param(
            *case,
            id=f"#{i}",
        )
        for i, case in enumerate(
            [
                ({}, "one two three", ["one", "two", "three"]),
                (
                    {},
                    """one   two\tthree four\t\tfive
            six

            seven""",
                    ["one", "two", "three", "four", "five", "six", "seven"],
                ),
                (
                    {"ngram_range": (1, 2)},
                    "one two three",
                    ["one", "two", "three", ("one", "two"), ("two", "three")],
                ),
                ({"ngram_range": (2, 2)}, "one two three", [("one", "two"), ("two", "three")]),
                (
                    {"ngram_range": (2, 3)},
                    "one two three",
                    [("one", "two"), ("two", "three"), ("one", "two", "three")],
                ),
                ({"stop_words": {"two", "three"}}, "one two three four", ["one", "four"]),
                (
                    {"stop_words": {"two", "three"}, "ngram_range": (1, 2)},
                    "one two three four",
                    ["one", "four", ("one", "four")],
                ),
            ]
        )
    ],
)
def test_ngrams(params, text, expected_ngrams):
    bow = feature_extraction.BagOfWords(**params)
    ngrams = list(bow.process_text(text))
    assert expected_ngrams == ngrams


CORPUS = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


def _assert_row_matches(one: dict, row) -> None:
    nonzero = {term: val for term, val in row.items() if val != 0}
    assert set(nonzero) == set(one)
    for term, weight in one.items():
        assert np.isclose(row[term], weight)


def test_tfidf_learn_many_matches_learn_one():
    batch = feature_extraction.TFIDF()
    batch.learn_many(pd.Series(CORPUS))
    stream = feature_extraction.TFIDF()
    for doc in CORPUS:
        stream.learn_one(doc)
    assert batch.dfs == stream.dfs
    assert batch.n == stream.n


def test_tfidf_transform_many_matches_transform_one_series():
    X = pd.Series(CORPUS)
    batch = feature_extraction.TFIDF()
    batch.learn_many(X)
    many = batch.transform_many(X)

    stream = feature_extraction.TFIDF()
    for doc in CORPUS:
        stream.learn_one(doc)
    for i, doc in enumerate(CORPUS):
        _assert_row_matches(stream.transform_one(doc), many.iloc[i])


def test_tfidf_transform_many_dataframe_with_on():
    # Exact repro from issue #1576: TFIDF(on="text").transform_many(df).
    df = pd.DataFrame({"text": CORPUS})
    batch = feature_extraction.TFIDF(on="text")
    batch.learn_many(df)
    many = batch.transform_many(df)  # must not raise
    assert len(many) == len(CORPUS)

    stream = feature_extraction.TFIDF(on="text")
    recs = df.to_dict(orient="records")
    for rec in recs:
        stream.learn_one(rec)
    for i, rec in enumerate(recs):
        _assert_row_matches(stream.transform_one(rec), many.iloc[i])


def test_tfidf_transform_many_without_normalize():
    X = pd.Series(CORPUS)
    batch = feature_extraction.TFIDF(normalize=False)
    batch.learn_many(X)
    many = batch.transform_many(X)
    stream = feature_extraction.TFIDF(normalize=False)
    for doc in CORPUS:
        stream.learn_one(doc)
    for i, doc in enumerate(CORPUS):
        _assert_row_matches(stream.transform_one(doc), many.iloc[i])


def test_bagofwords_transform_many_dataframe_with_on():
    # Regression for the shared root cause: previously raised TypeError.
    df = pd.DataFrame({"text": ["foo bar", "foo baz"]})
    out = feature_extraction.BagOfWords(on="text").transform_many(df)
    assert out.loc[0, "foo"] == 1 and out.loc[0, "bar"] == 1
    assert out.loc[1, "foo"] == 1 and out.loc[1, "baz"] == 1


def test_feature_extraction_estimators_pass_checks():
    from river import checks

    checks.check_estimator(feature_extraction.TFIDF())
    checks.check_estimator(feature_extraction.BagOfWords())
