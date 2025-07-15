# 🧾 Document Parser
AI 기반 문서 분석 시스템 – 다양한 형태의 문서(PDF, 이미지 등)에서 텍스트, 표, 레이아웃을 자동으로 추출하고 구조화합니다.

## 🔍 프로젝트 개요
이 프로젝트는 실무에서 발생하는 다양한 서류들을 자동으로 디지털화하고 분석하기 위해 만들어졌습니다.

- OCR 기반 텍스트 추출
- 문단 / 표 등 문서 레이아웃 인식
- 복잡한 테이블도 Graph 기반으로 구조 분석
- 실시간 시각화 및 후처리 기능 제공

## 🧑‍🤝‍🧑 개발자 소개 

- **이세연** : 팀장, Table 담당
- **서동진** : Chart 풀스택 담당
- **이재형** : Layout Parsing 담당
- **노성환** : OCR 담당


## ⚙️ 기술 스택

| 영역         |기술                                                                 |
|--------------|----------------------------------------------------------------------|
| 언어         | Python 3.8+                                                           |
| OCR          | doctr, google_ocr, easy_ocr, paddle_ocr         |
| Layout 분석  | docbank, layoutparser, pdfplumber, publaynet      |
| 이미지 처리   |                                                |
| 테이블 구조화 |                              |
| 시각화       |                               |

## 📂 프로젝트 구조

document-parser/
├── Layout_Parser/          # 문서 레이아웃 분석 모듈
├── OCR/                    # OCR 처리 모듈
├── document_parser/        # 실행 및 통합 로직 (메인 파서)
├── logger/                 # 로그 처리 및 디버깅 도구
├── results/                # 추출 결과 저장 디렉토리
├── sample_docs/            # 테스트용 샘플 문서
├── table/                  # 테이블 구조 분석 모듈
├── web/                    # 웹 UI 및 API 서버
├── .gitignore              # Git 무시 설정
├── LICENSE                 # 라이선스 파일


## API


## 📌 주요 기능
