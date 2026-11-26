import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents import (  # noqa: E402
    LLMConfigurationError,
    NeuroscienceAssistant,
    _require_llm_creds,
)


class LLMConfigurationErrorTests(unittest.TestCase):
    def test_vertex_mode_requires_project_id(self):
        with patch.dict(
            os.environ,
            {"GEMINI_USE_VERTEX": "true", "GCP_PROJECT_ID": ""},
            clear=True,
        ):
            with self.assertRaisesRegex(
                LLMConfigurationError,
                "GCP_PROJECT_ID must be set for Vertex mode",
            ):
                _require_llm_creds()

    def test_api_key_mode_requires_api_key(self):
        with patch.dict(
            os.environ,
            {"GEMINI_USE_VERTEX": "false", "GOOGLE_API_KEY": ""},
            clear=True,
        ):
            with self.assertRaisesRegex(
                LLMConfigurationError,
                "GOOGLE_API_KEY must be set for API-key mode",
            ):
                _require_llm_creds()

    def test_handle_chat_returns_generic_configuration_message(self):
        assistant = NeuroscienceAssistant()

        async def raise_configuration_error(*args, **kwargs):
            raise LLMConfigurationError("GCP_PROJECT_ID must be set for Vertex mode.")

        assistant.graph.ainvoke = raise_configuration_error

        import asyncio

        response = asyncio.run(
            assistant.handle_chat(session_id="test", query="hello")
        )

        self.assertEqual(
            response,
            "I'm sorry, my backend connection is currently misconfigured. "
            "Please contact the administrator.",
        )
        self.assertNotIn("GCP_PROJECT_ID", response)


if __name__ == "__main__":
    unittest.main()
