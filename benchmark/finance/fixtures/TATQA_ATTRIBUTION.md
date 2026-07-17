# TAT-QA attribution

The files tatqa_public_test_gold_cells.json and
../tool_routing_finance_tatqa_public_derived.json contain a narrow adaptation
of 15 records from the official TAT-QA test set with released gold answers.

## Source

- Dataset: TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and
  Textual Content in Finance
- Authors: Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang,
  Jiancheng Lv, Fuli Feng, and Tat-Seng Chua
- Official repository: https://github.com/NExTplusplus/TAT-QA
- Pinned revision: 870accc41953dcde885aabeb963d94aabdc0fbc3
- Source file:
  https://github.com/NExTplusplus/TAT-QA/blob/870accc41953dcde885aabeb963d94aabdc0fbc3/dataset_raw/tatqa_dataset_test_gold.json
- Source-file SHA-256:
  c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597
- Paper: https://aclanthology.org/2021.acl-long.254/
- DOI: https://doi.org/10.18653/v1/2021.acl-long.254

Suggested citation:

> Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng
> Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering
> Benchmark on a Hybrid of Tabular and Textual Content in Finance. In
> Proceedings of ACL-IJCNLP 2021, pages 3277–3287.

## License

The TAT-QA repository states that the dataset is licensed under the
[Creative Commons Attribution 4.0 International License][cc-by-4]. That license
permits sharing and adaptation, including commercial use, provided appropriate
credit is given, the license is linked, and modifications are indicated. The
repository's MIT license applies separately to its software.

[cc-by-4]: https://creativecommons.org/licenses/by/4.0/

## Changes made by LayerMCP

- Selected 15 table-sourced arithmetic questions from the official test-gold
  file; the question strings, answers, derivations, and source question
  metadata are retained unchanged.
- Retained every cell in each selected source table and added zero-based row
  and column coordinates.
- Added row and column labels derived from the source table layout.
- Added normalized numeric values alongside the exact original cell strings.
  Currency symbols and grouping punctuation are removed, parenthesized values
  are negative, displayed dashes are zero, and percentages remain in
  percentage-point units.
- Added bounded read-only SQL expressions implementing the published
  derivations for LayerMCP's finance table-query tool.

These adaptations do not imply endorsement by the TAT-QA authors.

