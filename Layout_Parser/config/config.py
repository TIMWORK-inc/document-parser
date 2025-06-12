class Config:
    ### 공통 경로
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUT_DIR = "outputs"
    LOG_DIR = "logs"

    ### 분석 대상 PDF 경로 (공통)
    DOCUMENT_PATH = f"{DATA_DIR}/KCS 14 20 11 철근공사.pdf"

    ### 공통 파라미터
    DPI = 300
    LOG_LEVEL = "INFO"

    ### LayoutParser 설정
    LAYOUT_PARSER = {
        "MODEL_PATH": f"{MODELS_DIR}/layoutparser_model.pth",
        "OUTPUT_JSON": f"{OUTPUT_DIR}/layoutparser_blocks.json",
        "OUTPUT_IMAGE_DIR": f"{OUTPUT_DIR}/visualized_pages_layoutparser",
        "SCORE_THRESH": 0.8,
        "LABEL_MAP": {
            0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"
        },
        "NUM_CLASSES": 5
    }

    ### PubLayNet 설정
    PUBLAYNET = {
        "MODEL_PATH": f"{MODELS_DIR}/publaynet_model.pth",
        "OUTPUT_JSON": f"{OUTPUT_DIR}/publaynet_blocks.json",
        "OUTPUT_IMAGE_DIR": f"{OUTPUT_DIR}/visualized_pages_publaynet",
        "SCORE_THRESH": 0.8,
        "LABEL_MAP": {
            0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"
        },
        "NUM_CLASSES": 5
    }

    ### DocBank 설정
    DOCBANK = {
        "MODEL_PATH": f"{MODELS_DIR}/docbank_model.pth",
        "OUTPUT_JSON": f"{OUTPUT_DIR}/docbank_blocks.json",
        "OUTPUT_IMAGE_DIR": f"{OUTPUT_DIR}/visualized_pages_docbank",
        "SCORE_THRESH": 0.6,
        "LABEL_MAP": {
            0: "abstract",
            1: "author",
            2: "caption",
            3: "date",
            4: "equation",
            5: "figure",
            6: "footer",
            7: "list",
            8: "paragraph",
            9: "reference",
            10: "section",
            11: "table",
            12: "title"
        },
        "NUM_CLASSES": 13
    }

    ### pdfplumber 설정
    PDFPLUMBER = {
        "OUTPUT_JSON": f"{OUTPUT_DIR}/pdfplumber_blocks.json",
        "OUTPUT_IMAGE_DIR": f"{OUTPUT_DIR}/visualized_pages_pdfplumber"
    }
