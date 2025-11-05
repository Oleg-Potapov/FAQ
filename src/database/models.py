from datetime import datetime, timezone
from sqlalchemy import Column, Integer, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from src.database.db import Base


class FaqStory(Base):
    __tablename__ = "faqstory"

    id = Column(Integer, primary_key=True)
    question = Column(String)
    response = Column(String)
    tokens = Column(JSONB, nullable=False)
    created = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


