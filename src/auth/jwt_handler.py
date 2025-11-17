"""
JWT token generation and validation.

This module provides secure JWT-based authentication for the API.
Features:
- Access token generation (30 minutes expiry)
- Refresh token generation (7 days expiry)
- Token validation and decoding
- Password hashing with bcrypt
- Role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# HTTP Bearer security scheme
security = HTTPBearer()


class JWTHandler:
    """
    Handles JWT token creation, validation, and password hashing.

    This class provides methods for:
    - Creating access and refresh tokens
    - Validating and decoding tokens
    - Hashing and verifying passwords
    """

    def __init__(self):
        """Initialize JWT handler with secret key from config."""
        self.secret_key = config.JWT_SECRET_KEY
        if not self.secret_key:
            logger.warning(
                "JWT_SECRET_KEY not set in environment. "
                "Using default (INSECURE - change in production!)"
            )
            self.secret_key = "INSECURE_DEFAULT_SECRET_KEY_CHANGE_IN_PRODUCTION"

    def create_access_token(
        self,
        user_id: str,
        email: str,
        roles: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Access tokens are short-lived (default 30 minutes) and used for
        authenticating API requests.

        Args:
            user_id: Unique user identifier
            email: User email address
            roles: List of user roles (e.g., ["user", "admin"])
            expires_delta: Custom token expiration time (optional)

        Returns:
            str: Encoded JWT token

        Example:
            >>> handler = JWTHandler()
            >>> token = handler.create_access_token(
            ...     user_id="123",
            ...     email="user@example.com",
            ...     roles=["user"]
            ... )
        """
        to_encode = {
            "sub": user_id,
            "email": email,
            "roles": roles,
            "type": "access"
        }

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow()
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        logger.info(f"Created access token for user {email} (ID: {user_id})")
        return encoded_jwt

    def create_refresh_token(self, user_id: str) -> str:
        """
        Create a refresh token for token renewal.

        Refresh tokens are long-lived (default 7 days) and used to obtain
        new access tokens without re-authentication.

        Args:
            user_id: Unique user identifier

        Returns:
            str: Encoded JWT refresh token

        Example:
            >>> handler = JWTHandler()
            >>> refresh_token = handler.create_refresh_token(user_id="123")
        """
        to_encode = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow()
        }
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        logger.info(f"Created refresh token for user ID: {user_id}")
        return encoded_jwt

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token.

        Args:
            token: JWT token string

        Returns:
            Dict[str, Any]: Decoded token payload containing user information

        Raises:
            HTTPException: If token is invalid, expired, or malformed

        Example:
            >>> handler = JWTHandler()
            >>> payload = handler.decode_token(token)
            >>> print(payload["sub"])  # User ID
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Token validation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=401,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Uses bcrypt for secure password verification.

        Args:
            plain_password: Password in plain text
            hashed_password: Bcrypt hashed password

        Returns:
            bool: True if password matches, False otherwise

        Example:
            >>> handler = JWTHandler()
            >>> is_valid = handler.verify_password("mypassword", hashed)
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}", exc_info=True)
            return False

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Password in plain text

        Returns:
            str: Bcrypt hashed password

        Example:
            >>> handler = JWTHandler()
            >>> hashed = handler.hash_password("mypassword")
        """
        return pwd_context.hash(password)

    def validate_token_type(self, token_payload: Dict[str, Any], expected_type: str) -> bool:
        """
        Validate that token is of expected type.

        Args:
            token_payload: Decoded token payload
            expected_type: Expected token type ("access" or "refresh")

        Returns:
            bool: True if token type matches

        Raises:
            HTTPException: If token type doesn't match
        """
        token_type = token_payload.get("type")
        if token_type != expected_type:
            logger.warning(
                f"Invalid token type. Expected: {expected_type}, Got: {token_type}"
            )
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token type. Expected {expected_type} token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True


# Global JWT handler instance
jwt_handler = JWTHandler()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency to extract and validate current user from JWT token.

    This dependency can be used in any route to require authentication.

    Args:
        credentials: HTTP Bearer credentials (automatically extracted)

    Returns:
        Dict[str, Any]: User information from token payload
            - sub: User ID
            - email: User email
            - roles: List of user roles
            - type: Token type ("access")

    Raises:
        HTTPException: If token is invalid or missing

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(user: dict = Depends(get_current_user)):
        >>>     return {"user_id": user["sub"], "email": user["email"]}
    """
    token = credentials.credentials
    payload = jwt_handler.decode_token(token)

    # Validate token type
    jwt_handler.validate_token_type(payload, "access")

    return payload


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current active user.

    This adds an additional check to ensure the user is active.
    In a full implementation, this would check user status in database.

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Dict[str, Any]: Active user information

    Raises:
        HTTPException: If user is inactive

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(user: dict = Depends(get_current_active_user)):
        >>>     return {"user_id": user["sub"]}
    """
    # In a full implementation, check user.is_active from database
    # For now, assume all users are active
    return current_user


def require_role(required_roles: List[str]):
    """
    Dependency factory for role-based access control.

    Creates a dependency that checks if user has one of the required roles.

    Args:
        required_roles: List of acceptable roles (e.g., ["admin", "moderator"])

    Returns:
        Callable: FastAPI dependency function

    Raises:
        HTTPException: If user doesn't have required role

    Example:
        >>> @app.post("/admin/action")
        >>> async def admin_action(
        >>>     user: dict = Depends(get_current_user),
        >>>     _: None = Depends(require_role(["admin"]))
        >>> ):
        >>>     return {"status": "success"}
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])

        # Check if user has at least one of the required roles
        if not any(role in user_roles for role in required_roles):
            logger.warning(
                f"User {current_user.get('sub')} lacks required roles: {required_roles}. "
                f"User roles: {user_roles}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {required_roles}",
            )

        return current_user

    return role_checker


def require_any_role(required_roles: List[str]):
    """
    Alias for require_role for clarity.

    Use when user needs at least one of the specified roles.
    """
    return require_role(required_roles)


def require_all_roles(required_roles: List[str]):
    """
    Dependency factory for requiring all specified roles.

    Creates a dependency that checks if user has ALL required roles.

    Args:
        required_roles: List of required roles (all must be present)

    Returns:
        Callable: FastAPI dependency function

    Raises:
        HTTPException: If user doesn't have all required roles

    Example:
        >>> @app.post("/super-admin/action")
        >>> async def super_admin_action(
        >>>     user: dict = Depends(get_current_user),
        >>>     _: None = Depends(require_all_roles(["admin", "superuser"]))
        >>> ):
        >>>     return {"status": "success"}
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])

        # Check if user has all required roles
        if not all(role in user_roles for role in required_roles):
            missing_roles = [role for role in required_roles if role not in user_roles]
            logger.warning(
                f"User {current_user.get('sub')} missing required roles: {missing_roles}. "
                f"User roles: {user_roles}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Missing roles: {missing_roles}",
            )

        return current_user

    return role_checker
