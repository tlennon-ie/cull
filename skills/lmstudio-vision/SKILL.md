# LMStudio Vision Integration Skill

**Purpose:** Call local LMStudio vision models for image classification and description generation.

**Models Supported:**
- Qwen-VL (recommended for accuracy)
- Gemma-Vision
- LLaVA

**Endpoint:** `http://127.0.0.1:8000/v1/` (configurable IP/port via environment or admin panel)

**API Format:** OpenAI-compatible (same as Groq cloud)

---

## Quick Start

```python
import requests
import base64

def assess_image_lmstudio(image_path: str, prompt: str = None) -> dict:
    """Call LMStudio for image assessment"""
    
    url = "http://127.0.0.1:8000/v1/chat/completions"
    
    # Read image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    payload = {
        "model": "qwen-vl",  # or gemma-vision, llava
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    },
                    {
                        "type": "text",
                        "text": prompt or "Classify this image as Professional, Amateur, or NSFW. Explain briefly."
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Call LMStudio
    response = requests.post(url, json=payload, timeout=60)
    result = response.json()
    
    # Extract response
    return {
        "classification": result["choices"][0]["message"]["content"],
        "model": result["model"],
        "usage": result["usage"]
    }
```

## Configuration

### Via Environment
```bash
LM_STUDIO_URL=http://127.0.0.1:8000
LMSTUDIO_MODEL=qwen-vl
LMSTUDIO_TIMEOUT=60  # seconds
```

### Via Python
```python
config = {
    "url": "http://127.0.0.1:8000",
    "model": "qwen-vl",
    "timeout": 60,
    "max_tokens": 500
}
```

### Via Admin Dashboard
1. Open http://localhost:5000
2. Admin → Vision Worker Config
3. Provider: LMStudio
4. Endpoint: http://127.0.0.1:8000 (change IP/port)
5. Model: Qwen-VL (dropdown)
6. Click Apply

## Switching IP/Port (Multiple LMStudio Instances)

If running LMStudio on different machine:

```bash
# Via environment
LM_STUDIO_URL=http://192.168.1.50:8000

# Via Python config
vision_config["lmstudio_url"] = "http://192.168.1.50:8000"

# Via CLI
python run_pipeline.py --lmstudio-url http://192.168.1.50:8000
```

## Error Handling

```python
try:
    result = assess_image_lmstudio(image_path)
except requests.ConnectionError:
    # LMStudio not running
    print("LMStudio unavailable at", url)
except requests.Timeout:
    # Model processing took too long
    print("LMStudio timeout (model overloaded)")
except Exception as e:
    # Other error
    print(f"LMStudio error: {e}")
    # Fall back to Groq or retry
```

## Classification Prompt Templates

### Quick (1 word)
```
"Classify: Professional, Amateur, or NSFW?"
```

### Standard (recommended)
```
"Classify this influencer image:
- Professional (studio, branded, polished)
- Amateur (user-generated, natural)
- NSFW (adult content)

Answer: [Classification]
Confidence: [0-100]
Why: [brief reason]"
```

### Detailed
```
"Analyze this image:
1. Classification: Professional/Amateur/NSFW
2. Quality score: 0-10
3. Key elements: list
4. Tags: comma-separated
5. Description: 1-2 sentences"
```

## Model Comparison

| Model | Speed | Accuracy | NSFW Detection | Best For |
|-------|-------|----------|----------------|----------|
| **Qwen-VL** | Medium | High | Excellent | General-purpose (recommended) |
| **Gemma-Vision** | Fast | Good | Good | Speed-focused batches |
| **LLaVA** | Medium | Medium | Okay | Fallback option |

## Health Check

```bash
# Test LMStudio connectivity
curl http://127.0.0.1:8000/health

# Test model loading
curl -X POST http://127.0.0.1:8000/v1/models

# Test inference (simple)
python -c "
from vision_lmstudio import assess_image_lmstudio
result = assess_image_lmstudio('test.jpg')
print('✅ LMStudio working:', result)
"
```

---

**See also:**
- `scripts/test_vision.py` — Testing script
- `references/qwen-vision-prompts.md` — Qwen-specific prompt templates
- `references/gemma-vision-prompts.md` — Gemma-specific templates
- Orchestrator.md → Vision Integration section
