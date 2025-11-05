import hashlib
import json
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.schemas import QuestionRequest, AskResponse, UploadDocumentsResponse, ListFaqStoryResponse
from src.core.config import OPENAI_API_KEY
from src.core.logger import setup_logger
from src.core.redis_client import redis_client
from src.core.utils import validate_and_normalize_question
from src.database.session import get_async_session
from src.repositories.faq_repositories import FaqRepository
from src.services.llm_service import LlmService

app = FastAPI()
logger = setup_logger(__name__)


@app.post("/api/documents", response_model=UploadDocumentsResponse)
async def upload_document(file: List[UploadFile] = File(...), db: AsyncSession = Depends(get_async_session)):
    try:
        llm_service = LlmService(FaqRepository(db), openai_api_key=OPENAI_API_KEY)
        result = await llm_service.process_uploaded_files(file)
        return UploadDocumentsResponse(root=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", response_model=AskResponse)
async def ask(question: QuestionRequest, db: AsyncSession = Depends(get_async_session)):
    try:
        # Валидация и нормализация (lower(), удаление лишних пробелов и опасных символов)
        normalized_question = validate_and_normalize_question(question.question)

        # Хешируем нормализованный вопрос для ключа Redis
        key_hash = hashlib.sha256(normalized_question.encode('utf-8')).hexdigest()

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Ошибка при валидации вопроса: {e}")
        raise HTTPException(status_code=400, detail="Некорректный формат вопроса")

    try:
        cached_answer = redis_client.get(key_hash)
        if cached_answer:
            logger.info(f"Cache hit for question hash {key_hash}")
            return {"answer": json.loads(cached_answer)}
        logger.info(f"Cache miss for question hash {key_hash}")
    except Exception as redis_error:
        logger.error(f"Ошибка при работе с Redis: {redis_error}")

    try:
        llm_service = LlmService(FaqRepository(db), openai_api_key=OPENAI_API_KEY)
        answer = await llm_service.answer_question(question.question)

        try:
            redis_client.setex(key_hash, 3600, json.dumps(answer))
            logger.info(f"Сохранён ответ в Redis с ключом {key_hash}")
        except Exception as redis_error:
            logger.error(f"Ошибка сохранения в Redis: {redis_error}")

        return {"answer": answer["answer"]}

    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")
# def ask(question: QuestionRequest, db: AsyncSession = Depends(get_async_session)):
#     if not question:
#         raise HTTPException(status_code=400, detail="Пустой вопрос")
#
#     question_text = question.question.strip().lower()
#     cached_answer = redis_client.get(question_text)
#
#     if cached_answer:
#         return {"answer": cached_answer}
#
#     try:
#         llm_service = LlmService(FaqRepository(db), openai_api_key=OPENAI_API_KEY)
#         answer = llm_service.answer_question(question.question)
#         redis_client.setex(question_text, 3600, answer)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")


@app.get("/api/story", response_model=ListFaqStoryResponse)
async def get_story(db: AsyncSession = Depends(get_async_session)):
    result = await FaqRepository(db).get_story_faq()
    return ListFaqStoryResponse(faqs=result)


@app.get("/api/health", response_model=Dict[str, str])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
