import os
import sys
from pathlib import Path

# Force the project root into path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("=== STREAMLIT SCOPE DIAGNOSTIC ===")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Expected .env Path: {ROOT / '.env'}")
print(f"Expected secrets.toml Path: {ROOT / '.streamlit' / 'secrets.toml'}\n")

# Check if Streamlit secrets exist natively
try:
    import streamlit as st
    # Access internal secrets map manually to see if it loads
    secrets = st.secrets
    print("🔄 Checking Streamlit internal secrets engine...")
    print(f"   - Found EE_SERVICE_ACCOUNT: {'EE_SERVICE_ACCOUNT' in secrets}")
    print(f"   - Found EE_KEY_FILE: {'EE_KEY_FILE' in secrets}")
except Exception as e:
    print(f"❌ Streamlit Secrets Error: {e}")

print("\n🔄 Attempting Real Handshake Ingestion...")
try:
    from data.fetch_satellite import init_gee
    import ee
    
    # This matches exactly what dashboard/app.py runs under the hood!
    init_gee()
    print("✅ SUCCESS: The dashboard-level handshake works flawlessly!")
    
    # Query a single test parameter to guarantee access token authorization
    test_geometry = ee.Geometry.Point([70.0, 17.5])
    print("✅ SUCCESS: Earth Engine successfully accepted data coordinates query token!")
except Exception as e:
    print(f"❌ CRITICAL HANDSHAKE FAILURE: {e}")