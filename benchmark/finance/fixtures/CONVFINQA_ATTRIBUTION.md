# ConvFinQA attribution

The multi-step finance fixture and benchmark use a narrow subset of the
development split released with:

> Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and
> William Yang Wang. "ConvFinQA: Exploring the Chain of Numerical Reasoning in
> Conversational Finance Question Answering." EMNLP 2022.

- Repository: <https://github.com/czyssrs/ConvFinQA>
- Pinned revision: `cf3eed2d5984960bf06bb8145bcea5e80b0222a6`
- Source archive: `data.zip`
- Archive SHA-256:
  `d764271fae60d81b62e6d58dfc481807ebc8cfbcd633811241723c4a2101072a`
- Source file inside the archive: `data/dev.json`
- License: MIT; the exact pinned license is preserved in
  `CONVFINQA_LICENSE.txt`.

The benchmark preserves the selected `dialogue_break` questions, gold
`turn_program` strings, execution answers, conversation IDs, row indexes, and
source filenames. The adaptation normalizes only cited gold evidence into a
bounded SQLite fixture and translates each gold program into the equivalent
arguments for LayerMCP's existing read-only table tool or calculator.
