import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

GOOGLE_OCR_API_KEY = os.getenv("GOOGLE_OCR_API_KEY")
