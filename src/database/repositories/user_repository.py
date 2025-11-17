"""
Repository for user data access.

This module provides methods for querying and managing users.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.database.models import User
from src.database.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for users."""

    def __init__(self, db: Session):
        super().__init__(User, db)

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_active(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users."""
        return self.db.query(User)\
            .filter(User.is_active == True)\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_role(self, role: str, skip: int = 0, limit: int = 100) -> List[User]:
        """Get users by role."""
        return self.db.query(User)\
            .filter(User.roles.contains([role]))\
            .offset(skip)\
            .limit(limit)\
            .all()

