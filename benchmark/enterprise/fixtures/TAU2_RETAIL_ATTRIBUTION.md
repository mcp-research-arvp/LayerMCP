# tau2 Retail Expansion Attribution

The retail single-step expansion is adapted from Sierra Research's tau2-bench
retail domain, pinned at
`363133ada1936491fb5bcec33cd62c3518a99f65` and distributed under the MIT
License.

Each row records the source task ID, action ID, train/test split, original gold
action arguments, and canonical source-task SHA-256. Raw tau2 tasks, database,
policy, and tool source remain outside this repository.

tau2 entity identifiers do not exist in LayerMCP's bounded retail fixture.
Rows therefore preserve the source gold action and intent while explicitly
mapping its entities to valid local fixture entities. Expected answers are
concise semantic subsets obtained from two identical gold executions with a
retail-state reset before each call.

