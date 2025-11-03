from pathlib import Path
import json

PROFILES_FILE = Path("profiles.json")

DEFAULTS = {
    "Default": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "price_input": 15000,
        "price_output": 30000,
        "avg_in": 600,
        "avg_out": 300,
        "rpm": 5,
    }
}

def _ensure():
    if not PROFILES_FILE.exists():
        PROFILES_FILE.write_text(json.dumps(DEFAULTS, indent=2))
        return DEFAULTS.copy()
    try:
        data = json.loads(PROFILES_FILE.read_text())
        if not isinstance(data, dict) or not data:
            raise ValueError("invalid structure")
        return data
    except Exception:
        PROFILES_FILE.write_text(json.dumps(DEFAULTS, indent=2))
        return DEFAULTS.copy()

def load_profiles():
    return _ensure()

def save_profiles(profiles: dict):
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2))

def list_profiles(profiles: dict | None = None):
    if profiles is None:
        profiles = load_profiles()
    return sorted(profiles.keys())

def load_profile(name: str):
    return load_profiles().get(name, DEFAULTS["Default"])

def save_profile(name: str, data: dict):
    profiles = load_profiles()
    profiles[name] = data
    save_profiles(profiles)

def delete_profile(name: str):
    profiles = load_profiles()
    profiles.pop(name, None)
    if not profiles:
        profiles = DEFAULTS.copy()
    save_profiles(profiles)
