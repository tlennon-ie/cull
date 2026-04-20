#!/usr/bin/env python3
"""
lmstudio_models.py — Fetch available models from LMStudio instances
Used by dashboard admin panel to let users select models at runtime
"""

import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PRIMARY_URL = os.getenv("LMSTUDIO_PRIMARY_URL", "http://100.75.25.43:1234")
SECONDARY_URL = os.getenv("LMSTUDIO_SECONDARY_URL", "http://100.95.148.26:1234")
TIMEOUT = 5

def fetch_models(url):
    """Fetch available models from LMStudio instance."""
    try:
        # LMStudio /v1/models endpoint
        response = requests.get(
            f"{url}/v1/models",
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            return [
                {
                    "id": m.get("id", "unknown"),
                    "name": m.get("id", "Unknown"),
                    "owned_by": m.get("owned_by", "lmstudio"),
                    "object": m.get("object", "model")
                }
                for m in models
            ]
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        print(f"Error fetching models from {url}: {e}")
        return None
    
    return []

def get_all_models():
    """Fetch models from both primary and secondary LMStudio."""
    result = {
        "primary": {
            "url": PRIMARY_URL,
            "models": [],
            "status": "unknown"
        },
        "secondary": {
            "url": SECONDARY_URL,
            "models": [],
            "status": "unknown"
        }
    }
    
    # Try primary
    models = fetch_models(PRIMARY_URL)
    if models is not None:
        result["primary"]["models"] = models
        result["primary"]["status"] = "connected" if models else "empty"
    else:
        result["primary"]["status"] = "disconnected"
    
    # Try secondary
    models = fetch_models(SECONDARY_URL)
    if models is not None:
        result["secondary"]["models"] = models
        result["secondary"]["status"] = "connected" if models else "empty"
    else:
        result["secondary"]["status"] = "disconnected"
    
    return result

def get_recommended_models():
    """Return recommended models (order of preference)."""
    return [
        {
            "id": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
            "name": "Qwen 3.5 9B (Aggressive Vision)",
            "description": "Best for fast inference, handles NSFW well"
        },
        {
            "id": "gemma-4-e4b",
            "name": "Gemma 4 E4B",
            "description": "Alternative vision model"
        },
        {
            "id": "qwen-vl",
            "name": "Qwen VL",
            "description": "Vision language model"
        }
    ]

if __name__ == "__main__":
    print("Fetching LMStudio models...")
    all_models = get_all_models()
    print(json.dumps(all_models, indent=2))
    
    print("\nRecommended models:")
    print(json.dumps(get_recommended_models(), indent=2))
