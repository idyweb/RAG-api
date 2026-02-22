from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception for API errors.
    Ensures clarity and actionable next steps."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An unexpected error occurred. Please try again.",
    ):
        super().__init__(status_code=status_code, detail=detail)


# ============== Authentication & Verification ==============


class InvalidCredentialsException(BaseAPIException):
    """Triggered when login fails. Simple and trustworthy."""

    def __init__(self, detail: str = "Incorrect phone number or password."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )


class OTPRequiredException(BaseAPIException):
    """Required for OTP verification flow."""

    def __init__(self, detail: str = "OTP verification required to continue."):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


class PhoneNotVerifiedException(BaseAPIException):
    def __init__(
        self,
        detail: str = "Please verify your phone number to access your energy dashboard.",
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


# ============== User & Asset Lifecycle ==============


class UserAlreadyExistsException(BaseAPIException):
    """Prevents duplicate registration by phone or email."""

    def __init__(
        self, detail: str = "An account with this phone number or email already exists."
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        )


class AssetNotFoundException(BaseAPIException):
    """Specific to internal asset management orchestration."""

    def __init__(self, detail: str = "Assigned energy asset not found."):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class EnergyRequestConflictException(BaseAPIException):
    """Prevents duplicate interest requests for the same site."""

    def __init__(
        self,
        detail: str = "You already have an active energy request for this location.",
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        )


# ============== Billing & Payments ==============


class PaymentFailedException(BaseAPIException):
    """Must be unambiguous for customers with low patience."""

    def __init__(
        self,
        detail: str = "Payment failed. Please check your balance or try a different method.",
    ):
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=detail,
        )


class BillingDiscrepancyException(BaseAPIException):
    """Supports transparency and trust principle."""

    def __init__(
        self,
        detail: str = "There is a discrepancy in your usage data. Our team is investigating.",
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


# ============== Staff & Permissions ==============


class PermissionDeniedException(BaseAPIException):
    """Enforces role-based access for Admin, Ops, and Finance staff."""

    def __init__(
        self, detail: str = "You do not have the required permissions for this action."
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


# ============== General Operational Exceptions ==============


class ResourceNotFoundException(BaseAPIException):
    """Generic fallback for missing resources."""

    def __init__(self, detail: str = "The requested information could not be found."):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


# ============== RAG & Documents ==============


class InvalidDepartmentError(BaseAPIException):
    """Triggered when an invalid department is provided for document ingestion."""

    def __init__(self, detail: str = "Invalid department provided."):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class DocumentNotFoundError(BaseAPIException):
    """Triggered when a requested document cannot be found."""

    def __init__(self, detail: str = "The requested document was not found."):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )