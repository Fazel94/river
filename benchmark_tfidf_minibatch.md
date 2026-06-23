# How fast is the mini-batch TFIDF?

I ran it against scikit-learn on the 20 Newsgroups train set (11,314 docs), both in
the same env with matching settings (no accent stripping, same token pattern, smooth
IDF + L2). Numbers are the best of 3 runs.

| what | output | time |
|---|---|---|
| sklearn `TfidfVectorizer.fit_transform` | CSR | 1.83 s |
| river `learn_many` (fit) | — | 1.58 s |
| river `_count_matrix` (tokenize + count → CSR) | CSR | 2.02 s |
| river `transform_many` (→ pandas sparse DF) | DataFrame | 3.57 s |
| river fit + transform | DataFrame | 4.59 s |

Both produce the same 11,314 × 112,842 matrix (1.1M non-zeros, ~0.09% dense), and the
values agree to about 1e-14 once the preprocessing matches.

So the actual vectorizing holds up fine — building the count matrix takes about as long
as sklearn's whole `fit_transform`. End to end we're roughly 2.5× slower, but that's
expected: the streaming API tokenizes twice (fit and transform are separate steps) and
hands back a pandas sparse DataFrame instead of a bare scipy matrix. For an online
library that's a fair trade.
