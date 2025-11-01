from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "question"}
        }
