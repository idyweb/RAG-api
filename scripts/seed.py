"""
Seed database script.

Populates the default IT Admin and test users across various departments.
"""

import asyncio
from api.db.database import async_session_factory
from api.apps.auth.models import User
from api.utils.security import hash_password
from api.utils.logger import get_logger

logger = get_logger(__name__)

# Test departments mapped to roles
USERS_TO_SEED = [
    {
        "email": "admin@coragem.com",
        "full_name": "System Admin",
        "password": "Password123!",
        "department": "IT",
        "role": "admin",
    },
    {
        "email": "hr.manager@coragem.com",
        "full_name": "HR Manager",
        "password": "Password123!",
        "department": "HR",
        "role": "manager",
    },
    {
        "email": "finance.rep@coragem.com",
        "full_name": "Finance Representative",
        "password": "Password123!",
        "department": "Finance",
        "role": "employee",
    },
    {
        "email": "sales.rep@coragem.com",
        "full_name": "Sales Rep",
        "password": "Password123!",
        "department": "Sales",
        "role": "employee",
    }
]


async def seed_users() -> None:
    async with async_session_factory() as session:
        try:
            logger.info("Starting database seed process...")
            
            for user_data in USERS_TO_SEED:
                # Check if user already exists
                existing = await User.find_unique(db=session, filters={"email": user_data["email"]})
                
                if existing:
                    logger.info(f"User {user_data['email']} already exists. Skipping.")
                    continue
                    
                logger.info(f"Creating user: {user_data['email']} in {user_data['department']}")
                
                hashed = hash_password(user_data["password"])
                
                user = User(
                    email=user_data["email"],
                    full_name=user_data["full_name"],
                    hashed_password=hashed,
                    department=user_data["department"],
                    role=user_data["role"],
                    is_active=True,
                    is_verified=True # Auto-verify seed users
                )
                
                session.add(user)
                
            await session.commit()
            logger.info("✅ Database seeded successfully!")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to seed database: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(seed_users())
