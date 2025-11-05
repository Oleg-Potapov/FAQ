from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class QuestionRequest(BaseModel):
    question: str = Field(..., example="Какой главный URL?")


class AskResponse(BaseModel):
    answer: str = Field(..., example="Главный URL компании EORA — это https://eora.com.")


class DocumentProcessResult(BaseModel):
    filename: str = Field(..., example="document.pdf")
    chunk_count: int = Field(..., example=10)
    message: str = Field(..., example="Документ успешно загружен и проиндексирован")


class UploadDocumentsResponse(BaseModel):
    root: list[DocumentProcessResult]
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "root": [
                    {
                        "filename": "SmartTask_User_Manual.pdf",
                        "chunk_count": 25,
                        "message": "Документ успешно загружен и проиндексирован"
                    },
                    {
                        "filename": "SmartTask_API_Guide.pdf",
                        "chunk_count": 15,
                        "message": "Документ успешно загружен и проиндексирован"
                    }
                ]
            }
        }
    )


class FaqStoryResponse(BaseModel):
    id: int
    question: str
    response: str
    tokens: dict
    created: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 123,
                "question": "Какой главный URL?",
                "response": "Главный URL компании EORA — это https://eora.com.",
                "tokens": {
                    "completion_tokens": 55,
                    "prompt_tokens": 56,
                    "total_tokens": 111,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0
                    },
                    "prompt_tokens_details": {
                        "audio_tokens": 0,
                        "cached_tokens": 0
                    }
                },
                "created": "2025-11-04T15:00:00Z"
            }
        }
    )


class ListFaqStoryResponse(BaseModel):
    faqs: List[FaqStoryResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "faqs": [
                    {
                        "id": 123,
                        "question": "Какой главный URL?",
                        "response": "Главный URL компании EORA — это https://eora.com.",
                        "tokens": {
                            "completion_tokens": 55,
                            "prompt_tokens": 56,
                            "total_tokens": 111,
                            "completion_tokens_details": {
                                "accepted_prediction_tokens": 0,
                                "audio_tokens": 0,
                                "reasoning_tokens": 0,
                                "rejected_prediction_tokens": 0
                            },
                            "prompt_tokens_details": {
                                "audio_tokens": 0,
                                "cached_tokens": 0
                            }
                        },
                        "created": "2025-11-04T15:00:00Z"
                    },
                    {
                        "id": 124,
                        "question": "Как работает сервис?",
                        "response": "Сервис работает по API OpenAI...",
                        "tokens": {
                            "completion_tokens": 45,
                            "prompt_tokens": 44,
                            "total_tokens": 89,
                            "completion_tokens_details": {},
                            "prompt_tokens_details": {}
                        },
                        "created": "2025-11-03T14:55:00Z"
                    }
                ]
            }
        }
    )
