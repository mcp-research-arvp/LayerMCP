import asyncio
from types import SimpleNamespace
import unittest

from evaluation.evaluate import load_tool_catalog, tool_pool_metadata


class EvaluationToolCatalogTests(unittest.TestCase):
    def test_public_catalog_preserves_order_and_baseline_registry_identity(self):
        tools = [SimpleNamespace(name=name, inputSchema={"type": "object"}, description=name)
                 for name in ("z_tool", "a_tool")]
        class Session:
            async def list_tools(self):
                return SimpleNamespace(tools=tools)
        catalog = asyncio.run(load_tool_catalog(Session()))
        self.assertEqual(catalog.names, ("z_tool", "a_tool"))
        self.assertEqual(catalog.metadata, tool_pool_metadata(list(catalog.names), catalog.schemas, catalog.descriptions))

    def test_duplicate_tool_names_are_rejected(self):
        class Session:
            async def list_tools(self):
                tool = SimpleNamespace(name="same", inputSchema={}, description="")
                return SimpleNamespace(tools=[tool, tool])
        with self.assertRaisesRegex(ValueError, "duplicate tool names"):
            asyncio.run(load_tool_catalog(Session()))
