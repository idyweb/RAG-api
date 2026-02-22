"""
Auth business logic.

Handles login, registration, and JWT-based user verification.
The `verify_user` function is the FastAPI dependency used by all secured routes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.apps.auth.models import User
from api.apps.auth.schemas import RegisterRequest, LoginRequest, UserResponse, TokenPair, LoginResponse
from api.db.session import get_session
from api.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token_type,
)
from api.utils.exceptions import PermissionDeniedException
from api.utils.logger import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer()


# ── FastAPI Auth Dependency ───────────────────────────────────────────────────

async def verify_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    FastAPI dependency: validates Bearer token and returns the active user dict.

    Used by all secured endpoints. Raises HTTP 401 if token is invalid or expired.
    Guards against inactive accounts.

    Returns:
        dict with: id, email, full_name, department, role
    """
    # Decode JWT — raises 401 on failure
    payload = verify_token_type(credentials.credentials, expected_type="access")
    user_id: str = payload.get("sub", "")

    # Fetch user from DB
    result = await session.execute(select(User).where(User.id == user_id))  # type: ignore
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    logger.debug(f"Authenticated user: {user.email} dept={user.department}")
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "department": user.department,
        "role": user.role,
    }


# ── Auth Services ─────────────────────────────────────────────────────────────

async def register_user(
    session: AsyncSession,
    data: RegisterRequest,
) -> UserResponse:
    """
    Create new employee account.

    Guard: Reject duplicate emails.
    Password is hashed before storage — never stored in plaintext.
    """
    # Guard: email uniqueness
    existing = await User.find_many(db=session, filters={"email": data.email}, limit=1)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await User.create(
        db=session,
        email=data.email,
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        department=data.department,
        role=data.role,
        is_active=True,
        is_verified=False,
    )

    logger.info(f"Registered new user: {user.email}, dept={user.department}")
    return UserResponse.model_validate(user)


async def login_user(
    session: AsyncSession,
    data: LoginRequest,
) -> LoginResponse:
    """
    Authenticate user and return JWT token pair.

    Guard: Reject bad credentials with generic error (no oracle attack).
    Guard: Reject inactive accounts.
    """
    # Fetch user
    users = await User.find_many(db=session, filters={"email": data.email}, limit=1)

    # Generic error — never reveal whether email exists
    if not users or not verify_password(data.password, users[0].hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = users[0]

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact IT.",
        )

    user_id = str(user.id)
    access_token = create_access_token(user_id=user_id, role=user.role, platform="web")
    refresh_token = create_refresh_token(user_id=user_id, role=user.role, platform="web")

    logger.info(f"User logged in: {user.email}")
    return LoginResponse(
        user=UserResponse.model_validate(user),
        tokens=TokenPair(access_token=access_token, refresh_token=refresh_token),
    )
