# tau2 Retail Expansion Attribution

The retail single-step expansion is adapted from Sierra Research's tau2-bench
retail domain, pinned at
`363133ada1936491fb5bcec33cd62c3518a99f65` and distributed under the MIT
License.

Each row records the source task ID, action index/ID, train/test split, original
gold action arguments, canonical source-task SHA-256, and the committed retail
fixture SHA-256.

LayerMCP packages a deterministically serialized derivative of tau2's pinned
`data/tau2/domains/retail/db.json` at
`mcp_server/fixtures/tau2_retail_db.json`. Native tau2 user, order, product,
item, and payment IDs are unchanged. The original and derived hashes,
transformation version, and upstream location are recorded in
`mcp_server/fixtures/tau2_retail_provenance.json`; the upstream MIT notice is
included beside the fixture.

Rows preserve native gold action arguments without entity remapping. Expected
answers are concise semantic subsets obtained from two identical executions
with a retail-state reset before each call. Actions that are not independently
executable from the initial pinned state are omitted.
