import os
import pickle
from typing import List

import chromadb
from fastapi import UploadFile
from openai import OpenAI
from chromadb import Client as ChromaClient
from chromadb.config import Settings
import PyPDF2


class LlmService:
    def __init__(self, openai_api_key,
                 cache_folder="cache",
                 chunk_size=500,
                 overlap=100):
        """
        Инициализация сервиса LLM:
        - openai_api_key: ключ OpenAI
        - cache_folder: папка для кэша данных
        - chunk_size: длина чанка текста (символы)
        - overlap: перекрытие между чанками (символы)
        """
        self.openai = OpenAI(api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        # Пути для кеша чанков
        self.chunk_texts_path = os.path.join(self.cache_folder, "chunk_texts.pkl")

        # Инициализация клиента ChromaDB с локальным API
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="faq_chunks")

        # Локальный кэш чанков для источников
        self.chunk_texts: List[str] = []
        self._load_chunks_cache()

    def _load_chunks_cache(self):
        """Загрузить чанки из локального кеша, если есть"""
        if os.path.isfile(self.chunk_texts_path):
            with open(self.chunk_texts_path, "rb") as f:
                self.chunk_texts = pickle.load(f)

    def _save_chunks_cache(self):
        """Сохранить локальный кеш чанков"""
        with open(self.chunk_texts_path, "wb") as f:
            pickle.dump(self.chunk_texts, f)

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Разбить текст на чанки с перекрытием.
        """
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    async def extract_text_from_pdf(self, file: UploadFile) -> str:
        """
        Извлечь текст из PDF файла с использованием PyPDF2.
        """
        file_bytes = await file.read()
        from io import BytesIO
        pdf_stream = BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    async def process_uploaded_file(self, file: UploadFile) -> dict:
        """
        Обработка загруженного файла: проверка формата,
        извлечение текста (pdf или текстовые файлы),
        создание чанков, эмбеддингов и добавление в ChromaDB.
        """
        filename = file.filename.lower()

        # Проверка расширения и извлечение текста
        if filename.endswith(".txt") or filename.endswith(".md"):
            content_bytes = await file.read()
            text = content_bytes.decode("utf-8")
        elif filename.endswith(".pdf"):
            text = await self.extract_text_from_pdf(file)
        else:
            raise ValueError("Поддерживаются файлы форматов .txt, .md и .pdf")

        if not text.strip():
            raise ValueError("Файл не содержит текста для обработки")

        # Разбиение на чанки
        chunks = self.split_text_into_chunks(text)

        # Генерация эмбеддингов через OpenAI API
        embeddings = []
        for chunk_text in chunks:
            response = self.openai.embeddings.create(
                input=chunk_text,
                model="text-embedding-3-large"
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)

        # Добавление чанков и эмбеддингов в коллекцию ChromaDB
        metadatas = [{"source": file.filename} for _ in chunks]
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Обновление локального кеша чанков
        self.chunk_texts.extend(chunks)
        self._save_chunks_cache()

        return {
            "filename": file.filename,
            "chunk_count": len(chunks),
            "message": "Документ успешно загружен и проиндексирован"
        }

    def _search_similar_chunks(self, query: str, top_k=5) -> List[str]:
        """
        Поиск релевантных чанков через эмбеддинги в ChromaDB.
        """
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response['data'][0]['embedding']

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        matched_chunks = results['documents'][0]
        return matched_chunks

    def answer_question(self, question: str, top_k=5) -> str:
        """
        Принимает вопрос, ищет релевантные чанки,
        формирует контекст и запрашивает ответ у LLM.
        """
        relevant_chunks = self._search_similar_chunks(question, top_k=top_k)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Ты помощник компании EORA. Используй следующий контекст из наших проектов и дай профессиональный,
полный и понятный ответ на вопрос пользователя.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

        chat_resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        answer = chat_resp.choices[0].message.content
        return answer
