from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import jwt
from jwt.exceptions import InvalidTokenError as JWTError
from fastapi import HTTPException, status
import secrets

# import bcrypt
from pwdlib import PasswordHash
import hashlib
import hmac
from user_agents import parse

from api.core.config import settings

password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a password using Argon2."""
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hashed password."""
    try:
        return password_hasher.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_token_hash(token: str) -> str:
    """Create a secure hash for Redis/DB storage using SHA-256 HMAC."""
    key = settings.JWT_SECRET_KEY.encode()
    return hmac.new(key, token.encode(), hashlib.sha256).hexdigest()


def create_access_token(
    user_id: str,
    role: str,
    platform: str,  # "mobile" or "web"
    session_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Web sessions may expire faster than Mobile for security
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    payload = {
        "sub": user_id,
        "role": role,
        "platform": platform,  # Essential for identifying Mobile vs Internal Web
        "sid": session_id,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def create_refresh_token(
    user_id: str, role: str, platform: str, session_id: Optional[str] = None
) -> str:
    """
    Create a JWT refresh token.
    Used for long-lived sessions on the Mobile App.
    """

    ttl_days = (
        settings.REFRESH_TOKEN_EXPIRE_DAYS_MOBILE
        if platform == "mobile"
        else settings.REFRESH_TOKEN_EXPIRE_DAYS_WEB
    )
    expire = datetime.now(timezone.utc) + timedelta(days=ttl_days)

    payload = {
        "sub": user_id,
        "role": role,
        "platform": platform,
        "sid": session_id,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "exp": expire,
    }
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def create_otp_token(user_id: str, phone_number: str) -> str:
    """
    Create a short-lived token specifically for OTP flows.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=10)
    payload = {
        "sub": user_id,
        "phone": phone_number,
        "type": "otp_verification",
        "exp": expire,
    }
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def verify_token_type(token: str, expected_type: str) -> dict[str, Any]:
    """Universal token decoder and type verifier."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {expected_type}",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify JWT token."""
    try:
        return jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        raise


def get_device_info(user_agent_str: str) -> dict[str, Any]:
    """
    Parse user agent.
    """
    user_agent = parse(user_agent_str)
    return {
        "os": user_agent.os.family,
        "device": user_agent.device.family,
        "is_mobile": user_agent.is_mobile,
        "is_pc": user_agent.is_pc,
        "app_platform": "mobile" if user_agent.is_mobile else "web",
    }


def generate_secure_string(length: int = 32) -> str:
    """Generate secure strings for OTPs or internal codes."""
    return secrets.token_urlsafe(length)