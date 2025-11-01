import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from src.api.schemas import QuestionRequest
from src.core.config import OPENAI_API_KEY
from src.services.llm_service import LlmService

app = FastAPI()


llm_service = LlmService(openai_api_key=OPENAI_API_KEY)


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    try:
        result = await llm_service.process_uploaded_file(file)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
def ask(question: QuestionRequest):
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    try:
        answer = llm_service.answer_question(question.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(e)}")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
