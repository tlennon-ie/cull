#!/usr/bin/env python3
import sys
import time

print("[TEST] Worker starting", flush=True)
time.sleep(2)
print("[TEST] Worker alive", flush=True)

# Try to import everything
try:
    from pathlib import Path
    print("[TEST] Path OK", flush=True)
    
    from PIL import Image
    print("[TEST] PIL OK", flush=True)
    
    import numpy as np
    print("[TEST] NumPy OK", flush=True)
    
    import json
    print("[TEST] JSON OK", flush=True)
    
    import io
    print("[TEST] IO OK", flush=True)
    
    import os
    print("[TEST] OS OK", flush=True)
    
    # Test queue dir
    BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
    QUEUE_DIR = BASE_DIR / "queue" / "realistic_female_influencer"
    print(f"[TEST] Queue dir exists: {QUEUE_DIR.exists()}", flush=True)
    
    # Test loop
    print("[TEST] Starting main loop", flush=True)
    for i in range(5):
        print(f"[TEST] Loop iteration {i}", flush=True)
        time.sleep(1)
    
    print("[TEST] Worker completed successfully", flush=True)
    
except Exception as e:
    print(f"[ERROR] {str(e)}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
