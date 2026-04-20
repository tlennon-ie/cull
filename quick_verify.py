#!/usr/bin/env python3
"""
Quick verification - no complex shell needed
"""
import os
import sys
import json
from pathlib import Path

# Set working directory
os.chdir(r'I:\AI\openclaw\workspace\claude\pipeline_code')
sys.path.insert(0, str(Path.cwd()))

from dotenv import load_dotenv
load_dotenv()

print("""
╔═══════════════════════════════════════════════════════════════╗
║     PIPELINE TEST: Configuration Verification                ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Test 1: Environment loading
print("\n✓ Test 1: Environment Loaded")
print(f"  PIPELINE_TOPIC: {os.getenv('PIPELINE_TOPIC')}")
print(f"  PIPELINE_SLUG: {os.getenv('PIPELINE_SLUG')}")
print(f"  LMSTUDIO_PRIMARY_URL: {os.getenv('LMSTUDIO_PRIMARY_URL')}")
print(f"  LMSTUDIO_SECONDARY_URL: {os.getenv('LMSTUDIO_SECONDARY_URL')}")

# Test 2: Discord configuration
print("\n✓ Test 2: Discord Configuration")
try:
    discord_json = os.getenv("DISCORD_CHANNELS_JSON", "{}")
    channels = json.loads(discord_json).get("channels", [])
    print(f"  Channels configured: {len(channels)}")
    for ch in channels:
        print(f"    - {ch.get('name')} ({ch.get('guild')})")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 3: Twitter cookies
print("\n✓ Test 3: Twitter Configuration")
twitter_cookies = os.getenv("TWITTER_COOKIES", "")
if twitter_cookies:
    print(f"  Cookies: ✅ Configured ({len(twitter_cookies)} chars)")
else:
    print(f"  Cookies: ❌ Not configured")

# Test 4: Civitai
print("\n✓ Test 4: Civitai Configuration")
civitai_domain = os.getenv("CIVITAI_DOMAIN", "civitai.com")
civitai_key = os.getenv("CIVITAI_API_KEY", "")
print(f"  Domain: {civitai_domain}")
print(f"  API Key: {'✅ Configured' if civitai_key else '❌ Not configured'}")

# Test 5: LMStudio connectivity
print("\n✓ Test 5: LMStudio Connectivity")
try:
    import requests
    
    primary_url = os.getenv("LMSTUDIO_PRIMARY_URL")
    secondary_url = os.getenv("LMSTUDIO_SECONDARY_URL")
    
    # Primary
    try:
        r = requests.get(f"{primary_url}/v1/models", timeout=2)
        if r.status_code == 200:
            models = r.json().get("data", [])
            print(f"  Primary ({primary_url}): ✅ Connected ({len(models)} models)")
        else:
            print(f"  Primary: ❌ HTTP {r.status_code}")
    except Exception as e:
        print(f"  Primary: ❌ Cannot connect ({str(e)[:30]})")
    
    # Secondary
    try:
        r = requests.get(f"{secondary_url}/v1/models", timeout=2)
        if r.status_code == 200:
            models = r.json().get("data", [])
            print(f"  Secondary ({secondary_url}): ✅ Connected ({len(models)} models)")
        else:
            print(f"  Secondary: ❌ HTTP {r.status_code}")
    except Exception as e:
        print(f"  Secondary: ❌ Cannot connect ({str(e)[:30]})")

except ImportError:
    print(f"  Requests library not available")

# Test 6: Queue structure
print("\n✓ Test 6: Queue Structure")
queue_base = Path(os.getenv("PIPELINE_QUEUE", "queue"))
slug = os.getenv("PIPELINE_SLUG", "realistic_female_influencer")
queue_path = queue_base / slug

print(f"  Base: {queue_base}")
print(f"  Topic: {slug}")
print(f"  Full path: {queue_path}")
print(f"  Status: {'✅ Exists' if queue_path.exists() else '❌ Does not exist'}")

# Test 7: Files exist
print("\n✓ Test 7: Required Files")
files_to_check = [
    "lmstudio_models.py",
    "dashboard_enhanced.py",
    "run_pipeline.py",
    "queue_manager.py",
    "scraper_discord.py",
    "scraper_civitai_search.py"
]

for f in files_to_check:
    exists = Path(f).exists()
    symbol = "✅" if exists else "❌"
    print(f"  {symbol} {f}")

print("\n" + "="*65)
print("READY TO RUN:\n")
print("  1. python test_harness.py          (Full test)")
print("  2. python dashboard_enhanced.py    (Dashboard on port 5000)")
print("  3. python run_pipeline.py          (Full pipeline)")
print("\n" + "="*65)
