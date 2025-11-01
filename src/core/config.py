from dotenv import load_dotenv
import os

load_dotenv()

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
