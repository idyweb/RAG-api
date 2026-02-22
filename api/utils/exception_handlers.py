from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException

from api.utils.logger import logger
from api.utils.exceptions import BaseAPIException


async def base_api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions and log them."""
    logger.error(
        f"{exc.__class__.__name__}: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "failure",
            "status_code": exc.status_code,
            "message": exc.detail,
            "error": {},
        },
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Handle validation errors and log them."""
    
    # Needs to be imported inside or at the top of file to use FastAPI JSON encoding
    from fastapi.encoders import jsonable_encoder
    
    errors = jsonable_encoder(exc.errors())
    logger.warning(
        f"Validation error on {request.method} {request.url.path}",
        extra={"errors": errors},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "failure",
            "status_code": 422,
            "message": "Validation error",
            "error": {"details": errors},
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException and log them."""
    logger.error(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "failure",
            "status_code": exc.status_code,
            "message": exc.detail,
            "error": {},
        },
        headers=exc.headers,
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions and log them."""
    logger.exception(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "failure",
            "status_code": 500,
            "message": str(exc),
            "error": {},
        },
    )