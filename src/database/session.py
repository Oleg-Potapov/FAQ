from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.db import async_session_maker


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
