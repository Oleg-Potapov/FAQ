import os
import pickle
import time
from typing import List
from fastapi import UploadFile, HTTPException
import PyPDF2
import chromadb
from openai import OpenAI
from src.core.logger import setup_logger
from src.core.utils import to_dict_recursive
from src.repositories.faq_repositories import FaqRepository
from rouge_score import rouge_scorer


class LlmService:
    def __init__(self, repository: FaqRepository, openai_api_key,
                 cache_folder="cache",
                 chunk_size=500,
                 overlap=100):
        self.reference_qa = {
            "Не приходят уведомления на email": "Проверьте спам и настройки уведомлений в профиле.",
            "Ошибка \"401 Unauthorized\"": "Проверьте действительность API ключа и формат заголовка.",
            "Не отображаются задачи в проекте": "Проверьте фильтры и права доступа.",
            "Приложение работает медленно": "Очистите кэш браузера, попробуйте другой браузер.",
            "Не удаётся загрузить файл": "Максимальный размер — 50 МБ. Проверьте соединение."
        }
        self.repository = repository
        self.logger = setup_logger(__name__)
        self.logger.info("Инициализация LlmService...")
        self.openai = OpenAI(api_key=openai_api_key)

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        self.chunk_texts_path = os.path.join(self.cache_folder, "chunk_texts.pkl")

        try:
            self.logger.info("Инициализация клиента ChromaDB...")
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name="faq_chunks")
            self.reference_collection = self.chroma_client.get_or_create_collection(name="reference_questions")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise HTTPException(status_code=500, detail="Ошибка инициализации БД")

        self._create_reference_embeddings()

        self.chunk_texts: List[str] = []
        self._load_chunks_cache()

    def _create_reference_embeddings(self):
        try:
            self.logger.info("Генерация и сохранение эмбеддингов эталонных вопросов в ChromaDB...")
            questions = list(self.reference_qa.keys())
            embeddings = []
            for question in questions:
                response = self.openai.embeddings.create(
                    input=question,
                    model="text-embedding-3-large"
                )
                embeddings.append(response.data[0].embedding)

            ids = [f"ref_question_{i}" for i in range(len(questions))]
            metadatas = [{"source": "reference_qa"} for _ in questions]

            # Очистка коллекции эталонных вопросов для предотвращения дублирования
            self.reference_collection.delete(where={"source": "reference_qa"})

            self.reference_collection.add(
                documents=questions,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Добавлено {len(questions)} эталонных вопросов в коллекцию ChromaDB")
        except Exception as e:
            self.logger.error(f"Ошибка при создании и сохранении эмбеддингов эталонных вопросов: {e}")

    def _load_chunks_cache(self):
        try:
            if os.path.isfile(self.chunk_texts_path):
                self.logger.info(f"Загружаю кеш чанков из {self.chunk_texts_path}")
                with open(self.chunk_texts_path, "rb") as f:
                    self.chunk_texts = pickle.load(f)
                self.logger.info(f"Загружено {len(self.chunk_texts)} чанков из кеша")
            else:
                self.logger.info("Кеш чанков отсутствует, загружать нечего")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки кеша чанков: {e}")

    def _save_chunks_cache(self):
        try:
            with open(self.chunk_texts_path, "wb") as f:
                pickle.dump(self.chunk_texts, f)
            self.logger.info(f"Сохранено {len(self.chunk_texts)} чанков в кеш по пути {self.chunk_texts_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения кеша чанков: {e}")

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
        try:
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
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении текста из PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Ошибка обработки PDF файла: {file.filename}")

    async def process_uploaded_files(self, files: List[UploadFile]) -> List[dict]:
        results = []
        for file in files:
            try:
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
                    raise HTTPException(status_code=415, detail="Поддерживаются файлы форматов .txt, .md и .pdf")

                if not text.strip():
                    self.logger.error("Файл не содержит текста для обработки")
                    raise HTTPException(status_code=400, detail="Файл не содержит текста для обработки")

                chunks = self.split_text_into_chunks(text)

                self.logger.info("Начинаю генерацию эмбеддингов для чанков через OpenAI API...")
                embeddings = []
                for i, chunk_text in enumerate(chunks):
                    self.logger.debug(f"Создаю эмбеддинг для чанка {i + 1}/{len(chunks)}")
                    try:
                        response = self.openai.embeddings.create(
                            input=chunk_text,
                            model="text-embedding-3-large"
                        )
                        embedding = response.data[0].embedding
                        embeddings.append(embedding)
                    except Exception as e:
                        self.logger.error(f"Ошибка при генерации эмбеддинга для чанка {i+1}: {e}")
                        raise HTTPException(status_code=502, detail="Ошибка связи с OpenAI API")

                self.logger.info(f"Сгенерировано {len(embeddings)} эмбеддингов")

                metadatas = [{"source": file.filename} for _ in chunks]
                ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]

                try:
                    self.logger.info(f"Добавляю чанки и эмбеддинги в коллекцию ChromaDB (количество: {len(chunks)})")
                    self.collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    self.logger.error(f"Ошибка добавления данных в ChromaDB: {e}")
                    raise HTTPException(status_code=500, detail="Ошибка записи в базу данных")

                self.chunk_texts.extend(chunks)
                self._save_chunks_cache()

                self.logger.info(f"Файл {file.filename} успешно проиндексирован")
                results.append({
                    "filename": file.filename,
                    "chunk_count": len(chunks),
                    "message": "Документ успешно загружен и проиндексирован"
                })
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Неожиданная ошибка при обработке файла {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "chunk_count": 0,
                    "message": f"Ошибка при обработке файла: {str(e)}"
                })
        return results

    def _search_similar_chunks(self, query: str, top_k=20) -> List[str]:
        try:
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
        except Exception as e:
            self.logger.error(f"Ошибка при поиске релевантных чанков: {e}")
            raise HTTPException(status_code=500, detail="Ошибка внутреннего сервиса поиска")

    def evaluate_answer(self, generated_answer: str, reference_answer: str) -> dict:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_answer, generated_answer)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }

    async def log_evaluation(self, question: str, generated_answer: str):
        try:
            response = self.openai.embeddings.create(
                input=question,
                model="text-embedding-3-large"
            )
            query_embedding = response.data[0].embedding

            results = self.reference_collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "distances"]
            )
            matched_questions = results['documents'][0]
            distances = results['distances'][0]

            if not matched_questions:
                self.logger.info("Похожий эталонный вопрос не найден для оценки.")
                return

            ref_question = matched_questions[0]
            ref_answer = self.reference_qa.get(ref_question)
            similarity = 1 - distances[0]

            metrics = self.evaluate_answer(generated_answer, ref_answer)

            self.logger.info(f"Оценка ответа на вопрос '{question}':")
            self.logger.info(f"Эталонный вопрос: '{ref_question}', Сходство: {similarity:.4f}")
            self.logger.info(f"Метрики ROUGE: {metrics}")

        except Exception as e:
            self.logger.error(f"Ошибка при логировании оценки: {e}")

    async def answer_question(self, question: str, top_k=5) -> dict:
        start_time = time.perf_counter()
        try:
            self.logger.info(f"Обработка вопроса: {question}")
            relevant_chunks = self._search_similar_chunks(question, top_k=top_k)
            context = "\n\n".join(relevant_chunks)

            prompt = f"""Ты — профессиональный помощник компании EORA, который отвечает на вопросы клиентов точно и понятно.

Используй исключительно информацию из следующего контекста. Не добавляй ничего лишнего.

Контекст:
{context}

Вопрос:
{question}

Дай полный, профессиональный и понятный ответ на вопрос.
Ответ:
"""

            self.logger.info("Запрос ответа у OpenAI Chat Completion...")
            chat_resp = self.openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            answer = chat_resp.choices[0].message.content
            usage = chat_resp.usage
            usage_dict = to_dict_recursive(usage) if usage else {}
            elapsed_time = (time.perf_counter() - start_time) * 1000
            self.logger.info(
                f"Получен ответ от модели OpenAI, токены: {usage}, "
                f"Время ответа: {elapsed_time:.2f} ms"
            )
            await self.repository.save_story_faq(question, answer, usage_dict)

            await self.log_evaluation(question, answer)

            return {
                "answer": answer,
                "usage": usage_dict,
            }
        except Exception as e:
            self.logger.error(f"Ошибка при обработке вопроса: {e}")
            raise

