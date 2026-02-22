"""
Auth router.

Entry/exit only — no logic here. Calls auth services.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.auth.schemas import RegisterRequest, LoginRequest, UserResponse, LoginResponse
from api.apps.auth.services import register_user, login_user, verify_user
from api.db.session import get_session
from api.utils.responses import success_response, auth_response

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", status_code=201)
async def register(
    data: RegisterRequest,
    session: AsyncSession = Depends(get_session),
):
    """Register a new employee. Returns user profile."""
    user = await register_user(session=session, data=data)
    return success_response(
        status_code=201,
        message="Account created successfully",
        data=user.model_dump(),
    )


@router.post("/login")
async def login(
    data: LoginRequest,
    session: AsyncSession = Depends(get_session),
):
    """Authenticate and receive JWT tokens."""
    result = await login_user(session=session, data=data)
    return auth_response(
        status_code=200,
        message="Login successful",
        access_token=result.tokens.access_token,
        refresh_token=result.tokens.refresh_token,
        data=result.user.model_dump(),
    )


@router.get("/me", response_model=UserResponse)
async def me(
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
):
    """Return the currently authenticated user's profile."""
    return success_response(
        status_code=200,
        message="User profile",
        data=user,
    )
