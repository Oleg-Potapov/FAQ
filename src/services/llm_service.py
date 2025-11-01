import os
import pickle
from typing import List
from fastapi import UploadFile
import PyPDF2
import chromadb
from openai import OpenAI
from src.core.logger import setup_logger


class LlmService:
    def __init__(self, openai_api_key,
                 cache_folder="cache",
                 chunk_size=500,
                 overlap=100):
        self.logger = setup_logger(__name__)
        self.logger.info("Инициализация LlmService...")
        self.openai = OpenAI(api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        self.chunk_texts_path = os.path.join(self.cache_folder, "chunk_texts.pkl")

        self.logger.info("Инициализация клиента ChromaDB...")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="faq_chunks")

        self.chunk_texts: List[str] = []
        self._load_chunks_cache()

    def _load_chunks_cache(self):
        if os.path.isfile(self.chunk_texts_path):
            self.logger.info(f"Загружаю кеш чанков из {self.chunk_texts_path}")
            with open(self.chunk_texts_path, "rb") as f:
                self.chunk_texts = pickle.load(f)
            self.logger.info(f"Загружено {len(self.chunk_texts)} чанков из кеша")
        else:
            self.logger.info("Кеш чанков отсутствует, загружать нечего")

    def _save_chunks_cache(self):
        with open(self.chunk_texts_path, "wb") as f:
            pickle.dump(self.chunk_texts, f)
        self.logger.info(f"Сохранено {len(self.chunk_texts)} чанков в кеш по пути {self.chunk_texts_path}")

    def split_text_into_chunks(self, text: str) -> List[str]:
        self.logger.info(f"Разбиение текста длиной {len(text)} символов на чанки размером {self.chunk_size} с перекрытием {self.overlap}")
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        self.logger.info(f"Создано {len(chunks)} чанков")
        return chunks

    async def extract_text_from_pdf(self, file: UploadFile) -> str:
        self.logger.info(f"Извлечение текста из PDF файла: {file.filename}")
        file_bytes = await file.read()
        from io import BytesIO
        pdf_stream = BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(pdf_stream)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                self.logger.debug(f"Страница {i} PDF пуста или не содержит текста")
        self.logger.info(f"Извлечено {len(text)} символов текста из PDF")
        return text

    async def process_uploaded_files(self, files: List[UploadFile]) -> List[dict]:
        results = []
        for file in files:
            self.logger.info(f"Обработка загруженного файла: {file.filename}")
            filename = file.filename.lower()

            if filename.endswith(".txt") or filename.endswith(".md"):
                content_bytes = await file.read()
                text = content_bytes.decode("utf-8")
                self.logger.info(f"Извлечен текст из текстового файла, длина {len(text)}")
            elif filename.endswith(".pdf"):
                text = await self.extract_text_from_pdf(file)
            else:
                self.logger.error(f"Неподдерживаемый формат файла: {filename}")
                raise ValueError("Поддерживаются файлы форматов .txt, .md и .pdf")

            if not text.strip():
                self.logger.error("Файл не содержит текста для обработки")
                raise ValueError("Файл не содержит текста для обработки")

            chunks = self.split_text_into_chunks(text)

            self.logger.info("Начинаю генерацию эмбеддингов для чанков через OpenAI API...")
            embeddings = []
            for i, chunk_text in enumerate(chunks):
                self.logger.debug(f"Создаю эмбеддинг для чанка {i + 1}/{len(chunks)}")
                response = self.openai.embeddings.create(
                    input=chunk_text,
                    model="text-embedding-3-large"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            self.logger.info(f"Сгенерировано {len(embeddings)} эмбеддингов")

            metadatas = [{"source": file.filename} for _ in chunks]
            ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

            self.logger.info(f"Добавляю чанки и эмбеддинги в коллекцию ChromaDB (количество: {len(chunks)})")
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            self.chunk_texts.extend(chunks)
            self._save_chunks_cache()

            self.logger.info(f"Файл {file.filename} успешно проиндексирован")
            results.append({
                "filename": file.filename,
                "chunk_count": len(chunks),
                "message": "Документ успешно загружен и проиндексирован"
            })
        return results

    def _search_similar_chunks(self, query: str, top_k=5) -> List[str]:
        self.logger.info(f"Поиск релевантных чанков для запроса: {query}")
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        matched_chunks = results['documents'][0]
        self.logger.info(f"Найдено {len(matched_chunks)} релевантных чанков")
        return matched_chunks

    def answer_question(self, question: str, top_k=5) -> str:
        self.logger.info(f"Обработка вопроса: {question}")
        relevant_chunks = self._search_similar_chunks(question, top_k=top_k)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Ты помощник компании EORA. Используй следующий контекст из наших проектов и дай профессиональный,
полный и понятный ответ на вопрос пользователя.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

        self.logger.info("Запрос ответа у OpenAI Chat Completion...")
        chat_resp = self.openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        answer = chat_resp.choices[0].message.content
        self.logger.info("Получен ответ от модели OpenAI")
        return answer
