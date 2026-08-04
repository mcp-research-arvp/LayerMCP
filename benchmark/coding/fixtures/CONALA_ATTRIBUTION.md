# CoNaLa public-derived coding fixture

## Source and citation

This fixture uses exact crowd-rewritten intents from the curated test split of
the author-affiliated [NeuLab CoNaLa dataset repository](https://huggingface.co/datasets/neulab/conala)
at revision `fbc749f1c537e5c3834e93b15784302e331debe2`:

- [`data/conala-paired-test.json`](https://huggingface.co/datasets/neulab/conala/blob/fbc749f1c537e5c3834e93b15784302e331debe2/data/conala-paired-test.json),
  SHA-256 `3a7e5eea6deeccb5e7c9557534af860854fd2f0ae870752b42c296ed30e53cb7`;
- the pinned [dataset card](https://huggingface.co/datasets/neulab/conala/blob/fbc749f1c537e5c3834e93b15784302e331debe2/README.md),
  SHA-256 `326072b41743fff642a4639ade350308a47942ff67608f3dfe447014453f3e74`;
- the pinned [dataset loader](https://huggingface.co/datasets/neulab/conala/blob/fbc749f1c537e5c3834e93b15784302e331debe2/conala.py),
  SHA-256 `1f106c699e97915f6d02b4f3a169a33f6de8e7306cc30b10cc1354d4b86f0f2d`.

The loader identifies the release as CoNaLa `1.1.0` and maps the curated test
split to the pinned source file. The associated paper is:

> Pengcheng Yin, Bowen Deng, Edgar Chen, Bogdan Vasilescu, and Graham Neubig.
> “Learning to Mine Aligned Code and Natural Language Pairs from Stack
> Overflow.” MSR 2018.

Paper: <https://arxiv.org/abs/1805.08949>

DOI: <https://doi.org/10.1145/3196398.3196408>

## Selection and normalization

The source contains 500 curated test examples. The deterministic selection
scans them in source order and retains a non-empty, single-line
`rewritten_intent` only when neither it nor an already selected intent is a
substring of the other. Selection stops after 133 records. This makes every
case-sensitive fixed-string query line-unique and brings LayerMCP's active
Coding single-step total from 167 to exactly 300. The selection rule is
versioned as `conala_curated_test_line_unique_133_v1`.

The selected intents come from 102 Stack Overflow questions and have 132
distinct paired-snippet hashes. Each fixture record retains the zero-based
source row index, Stack Overflow
question ID and derived question URL, exact `rewritten_intent`, canonical source
record SHA-256, and paired snippet SHA-256. Original Stack Overflow titles and
Python snippets are not copied. The full pinned source hash and per-record
hashes allow regeneration to verify the omitted source fields.

The benchmark `query` is a generated, self-contained repository-search
instruction. The exact CoNaLa text remains unchanged in `original_query`. This
distinction is recorded through `query_origin`, `original_query_origin`, and a
versioned `query_wrapper_id`.

## License and scope

The pinned dataset card declares `mit` under `license` in its YAML front matter.
LayerMCP records this as dataset-scoped license metadata. The pinned release
does not include a standalone dataset license notice, so no license text is
copied into this fixture.

This adaptation evaluates coding-tool routing and bounded, case-sensitive
fixed-string lookup. It does not reproduce CoNaLa code generation, its full
curated test split, tokenization, or BLEU evaluation. It also does not claim
that the generated search wrapper is a CoNaLa task.
