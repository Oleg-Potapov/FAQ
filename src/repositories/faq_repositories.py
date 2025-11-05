from abc import ABC, abstractmethod
from typing import List

from fastapi import HTTPException
from sqlalchemy import select, desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import FaqStory


class AbstractFaqRepository(ABC):
    @abstractmethod
    async def get_story_faq(self, limit: int = 20):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    @abstractmethod
    async def save_story_faq(self, question: str, response: str, tokens: dict):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")


class FaqRepository(AbstractFaqRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    # async def get_story_faq(self) -> FaqStory:
    #     try:
    #         result = await self.session.execute(select(FaqStory))
    #         rates = result.scalars().first()
    #         return rates
    #     except SQLAlchemyError as e:
    #         raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_story_faq(self, limit: int = 20) -> List[FaqStory]:
        try:
            stmt = (
                select(FaqStory)
                .order_by(desc(FaqStory.created))  # сортировка последних по дате создания
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            records = list(result.scalars().all())
            return records
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def save_story_faq(self, question: str, response: str, tokens: dict) -> FaqStory:
        try:
            new_record = FaqStory(question=question, response=response, tokens=tokens)
            self.session.add(new_record)
            await self.session.commit()
            await self.session.refresh(new_record)
            return new_record
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
