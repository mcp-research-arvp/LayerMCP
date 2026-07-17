# CodeSearchNet public-derived coding fixture

## Source and citation

This fixture uses selected published query and annotation records from the
[CodeSearchNet repository](https://github.com/github/CodeSearchNet) at the
human-evaluation publication commit
`bb121a53a559e99a6849409355ee5c83803f2e87`:

- [`resources/queries.csv`](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/resources/queries.csv), SHA-256 `037509c717c2e164721f0fd3ea45cb05f36669551af643f53930a92b76b146cf`.
- [`resources/annotationStore.csv`](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/resources/annotationStore.csv), SHA-256 `0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846`.
- [`LICENSE`](https://github.com/github/CodeSearchNet/blob/bb121a53a559e99a6849409355ee5c83803f2e87/LICENSE), preserved locally as [`CODESEARCHNET_LICENSE.txt`](CODESEARCHNET_LICENSE.txt), SHA-256 `5ba1fd8a344040f2698ed3234aeb8f4b3e85211aa54a37048021f3eb0043be22`.

The same query and annotation bytes were independently verified against
repository tip `106e827405c968597da938f6b373d30183918869`.

The associated paper is:

> Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
> Brockschmidt. “CodeSearchNet Challenge: Evaluating the State of Semantic
> Code Search.” arXiv:1909.09436, 2019.

Paper: <https://arxiv.org/abs/1909.09436>

## Selection and normalization

The fixture selects 15 exact CodeSearchNet query strings. Each is paired with
one selected Python annotation whose published relevance is `3`. All source
indices are explicitly zero-based and exclude the CSV header. For every
selected pair, the query text matches byte-for-byte and the
`(Language, Query, GitHubUrl)` multiplicity in the pinned annotation file is
exactly one. The complete selected annotation record, including its original
field names, values, URL, rating, and notes, is retained in the fixture.

CodeSearchNet publishes these queries, but the source does not expose
per-query authorship or establish that each query was authored in the paper.
Accordingly, benchmark metadata marks `original_query_origin` as
`codesearchnet_published_query`, not as a paper-authored question. The benchmark
`query` field is explicitly marked as a generated instruction wrapper; the
source text itself remains unchanged in `original_query` and in this fixture.

## License and scope

The CodeSearchNet repository, query file, and annotation file are distributed
under the MIT License. The exact pinned copyright and permission notice is
preserved in `CODESEARCHNET_LICENSE.txt`.

No target repository source code referenced by the annotation URLs is copied
into this fixture. The URLs are retained only as published annotation evidence
and provenance. The declarative repository normalizes only the selected
MIT-licensed CodeSearchNet query and annotation records.

This adaptation evaluates coding-tool routing and bounded, case-sensitive
fixed-string annotation lookup. It does not reproduce CodeSearchNet semantic
code retrieval, the full human relevance pool, ranking, or NDCG evaluation.
