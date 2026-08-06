import unittest

from analysis.rescore_saved_evaluation import rescore_records
from mcp_server.retail_state import snapshot_retail_state


class RescoreSavedEvaluationTests(unittest.TestCase):
    def test_rescores_saved_finance_table_result_with_column_alias(self) -> None:
        records, changed = rescore_records(
            [
                {
                    "sample_id": "finance-alias",
                    "domain": "finance",
                    "expected_tool": "finance_query_table",
                    "called_tool": "finance_query_table",
                    "expected_answer": {
                        "dataset_id": "fixture",
                        "columns": ["result"],
                        "rows": [[0.25]],
                        "row_count": 1,
                        "truncated": False,
                    },
                    "tool_result_value": {
                        "dataset_id": "fixture",
                        "columns": ["numeric_result"],
                        "rows": [[0.25]],
                        "row_count": 1,
                        "truncated": False,
                    },
                    "execution_success": True,
                    "final_outcome_correct": False,
                }
            ]
        )

        self.assertEqual(changed, ["finance-alias"])
        self.assertTrue(records[0]["final_outcome_correct"])

    def test_replays_saved_unprefixed_retail_order_lookup(self) -> None:
        order_id = next(iter(snapshot_retail_state()["orders"]))
        self.assertTrue(order_id.startswith("#W"))
        records, changed = rescore_records(
            [
                {
                    "sample_id": "retail-order-id",
                    "domain": "enterprise",
                    "expected_tool": "get_order_details",
                    "called_tool": "get_order_details",
                    "selected_args": {"order_id": order_id[1:]},
                    "expected_answer": {"order_id": order_id},
                    "tool_result_value": None,
                    "execution_success": False,
                    "final_outcome_correct": False,
                }
            ],
            replay_retail_order_lookups=True,
        )

        self.assertEqual(changed, ["retail-order-id"])
        self.assertTrue(records[0]["execution_success"])
        self.assertTrue(records[0]["final_outcome_correct"])
        self.assertEqual(records[0]["tool_result_value"]["order_id"], order_id)
