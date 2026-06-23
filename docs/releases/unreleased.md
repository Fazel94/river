# Unreleased

## feature_extraction

- `feature_extraction.TFIDF` now implements the mini-batch API: `learn_many` updates document frequencies online and `transform_many` returns TF-IDF–weighted sparse output, matching `learn_one`/`transform_one`. Fixes [#1576](https://github.com/online-ml/river/issues/1576).
- `feature_extraction.BagOfWords` and `feature_extraction.TFIDF` are now `MiniBatchTransformer`s. `transform_many` accepts a `pandas.DataFrame` when `on` is set (previously raised `TypeError`).
