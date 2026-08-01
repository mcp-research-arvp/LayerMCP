# SWE-agent / SWE-bench trajectory attribution

This fixture uses the official demonstration trajectory referenced by the
SWE-agent documentation for SWE-bench instance
`marshmallow-code__marshmallow-1867`.

- Paper: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret,
  Shunyu Yao, Karthik Narasimhan, and Ofir Press. "SWE-agent:
  Agent-Computer Interfaces Enable Automated Software Engineering."
  NeurIPS 2024.
- SWE-agent repository: <https://github.com/SWE-agent/SWE-agent>
- Pinned SWE-agent revision:
  `3ea751c087f32b16e039a2233dd6eefecef325d5`
- Trajectory SHA-256:
  `8856076ec31832f20aefa7f0a2714e3ad6bc752f14815d94d2e852e50213a459`
- SWE-bench instance: `marshmallow-code__marshmallow-1867`
- Upstream repository: <https://github.com/marshmallow-code/marshmallow>
- Upstream base commit:
  `bfd2593d4b416122e30cdefe0c72d322ef471611`
- Gold pull request: <https://github.com/marshmallow-code/marshmallow/pull/1867>
- License: MIT.

The original trajectory has 14 actions, including installation, temporary file
creation, execution, editing, cleanup, and submission. LayerMCP's coding tools
are intentionally read-only, so this fixture preserves the three reusable
repository-exploration actions exactly: root listing, locating `fields.py`,
and opening its relevant source window. It does not claim to reproduce the
trajectory's mutation or test-execution stages.
