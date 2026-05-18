import os
from dotenv import load_dotenv

load_dotenv()

def env(name: str, default=None, required: bool=False):
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v
