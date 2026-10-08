import os
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import hmac
import hashlib
import base64
import json

# --- Local Auth Config (from environment) ---
LOCAL_SECRET_KEY = os.getenv("LOCAL_SECRET_KEY", "change-me-please-use-a-random-secret")
LOCAL_USERNAME = os.getenv("LOCAL_USERNAME", "admin")
LOCAL_PASSWORD = os.getenv("LOCAL_PASSWORD", "password123")

TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

# ---------------------------------------------------------------------------
# Minimal "signed token" implementation — no third-party JWT library needed.
# Format:  base64(header).base64(payload).base64(HMAC-SHA256 signature)
# ---------------------------------------------------------------------------

def _b64_encode(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode().rstrip("=")

def _b64_decode(s: str) -> dict:
    # Restore padding
    padding = 4 - len(s) % 4
    s += "=" * (padding % 4)
    return json.loads(base64.urlsafe_b64decode(s))

def _sign(header_b64: str, payload_b64: str) -> str:
    msg = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(LOCAL_SECRET_KEY.encode(), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")

def create_token(username: str) -> str:
    header = _b64_encode({"alg": "HS256", "typ": "JWT"})
    exp = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)).timestamp()
    payload = _b64_encode({"sub": username, "exp": exp})
    signature = _sign(header, payload)
    return f"{header}.{payload}.{signature}"

def verify_token(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token")
        header_b64, payload_b64, sig_b64 = parts
        expected_sig = _sign(header_b64, payload_b64)
        if not hmac.compare_digest(sig_b64, expected_sig):
            raise ValueError("Invalid signature")
        payload = _b64_decode(payload_b64)
        if datetime.now(timezone.utc).timestamp() > payload["exp"]:
            raise ValueError("Token expired")
        return payload
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    return verify_token(credentials.credentials)
