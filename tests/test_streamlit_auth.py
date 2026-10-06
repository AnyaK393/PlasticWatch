import os
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestStreamlitAuth(unittest.TestCase):
    """Diagnostic check for Streamlit secrets and Earth Engine handshake."""

    def test_streamlit_secrets_or_skip(self):
        try:
            import streamlit as st
            secrets = getattr(st, "secrets", {})
            has_sa = "EE_SERVICE_ACCOUNT" in secrets
            has_key = "EE_KEY_FILE" in secrets
        except Exception:
            has_sa, has_key = False, False

        has_env = bool(os.getenv("EE_SERVICE_ACCOUNT") or os.getenv("EE_PROJECT"))
        if not (has_sa or has_env):
            self.skipTest("Earth Engine credentials not configured in secrets.toml or environment.")

        from data.fetch_satellite import init_gee
        init_gee()


if __name__ == "__main__":
    unittest.main()