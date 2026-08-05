# tau2 Retail Expansion Attribution

The retail single-step and reference-workflow expansions are adapted from
Sierra Research's tau2-bench retail domain, pinned at
`363133ada1936491fb5bcec33cd62c3518a99f65` and distributed under the MIT
License.

Each row records the source task ID, action index/ID, train/test split, original
reference action arguments, canonical source-task SHA-256, and the committed
retail fixture SHA-256.

LayerMCP packages a deterministically serialized derivative of tau2's pinned
`data/tau2/domains/retail/db.json` at
`mcp_server/fixtures/tau2_retail_db.json`. Native tau2 user, order, product,
item, and payment IDs are unchanged. The original and derived hashes,
transformation version, and upstream location are recorded in
`mcp_server/fixtures/tau2_retail_provenance.json`; the upstream MIT notice is
included beside the fixture.

Single-step rows preserve native reference action arguments without entity
remapping. Their expected answers are concise semantic subsets obtained from
two identical executions with a retail-state reset before each call. Actions
that are not independently executable from the initial pinned state are
omitted.

`enterprise_public_workflows.json` preserves fully supported multi-action
reference trajectories from tau2 evaluation criteria. These actions are one
released route to the target database state, not a uniquely correct plan; none
of the included workflows uses `ACTION` in `reward_basis`. LayerMCP evaluates
them as teacher-forced reference-trajectory routing, not tau2 task success.
Each predicted step executes in fresh retail state after replaying earlier
reference actions as state setup. Primitive expected results use FastMCP's
structured `{"result": ...}` representation.
