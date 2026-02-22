from typing import Optional, Any
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def success_response(
    status_code: int, message: str, data: Optional[dict | list | Any] = None
):
    """Returns a JSON response for success responses"""

    response_data = {
        "status": "success",
        "status_code": status_code,
        "message": message,
        "data": data or {},
    }

    return JSONResponse(
        status_code=status_code, content=jsonable_encoder(response_data)
    )


def auth_response(
    status_code: int,
    message: str,
    access_token: str,
    refresh_token: str,
    data: Optional[dict] = None,
):
    """Returns a JSON response for successful auth responses"""

    response_data = {
        "status": "success",
        "status_code": status_code,
        "message": message,
        "data": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            **(data or {}),
        },
    }

    return JSONResponse(
        status_code=status_code, content=jsonable_encoder(response_data)
    )


def fail_response(status_code: int, message: str, context: Optional[dict] = None):
    """Returns a JSON response for failure responses"""

    response_data = {
        "status": "failure",
        "status_code": status_code,
        "message": message,
        "error": context or {},
    }

    return JSONResponse(
        status_code=status_code, content=jsonable_encoder(response_data)
    )


def validation_error_response(errors: dict):
    """Standardized validation error response"""

    response = {
        "error": "VALIDATION_ERROR",
        "message": "The request contains invalid fields",
        "status_code": 422,
        "errors": errors,
    }

    return JSONResponse(status_code=422, content=jsonable_encoder(response))