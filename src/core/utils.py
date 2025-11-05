import re
from fastapi import HTTPException


def validate_and_normalize_question(text: str, max_length=500) -> str:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) > max_length:
        raise HTTPException(status_code=400, detail=f"Вопрос не должен превышать {max_length} символов")
    # Удаляем управляющие символы
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Приводим к нижнему регистру
    return text.lower()


def to_dict_recursive(obj):
    if isinstance(obj, dict):
        return {k: to_dict_recursive(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {k: to_dict_recursive(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [to_dict_recursive(i) for i in obj]
    else:
        return obj
