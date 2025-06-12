import os
import json
import pdfplumber
import cv2
import numpy as np

from config.config import Config
from utils.data_schema import BaseDetectionResult

# 새 Logger 시스템 import
from Logger import LayoutDetectionLogger

class PdfPlumberAdapter:
    def __init__(self):
        LayoutDetectionLogger.info("PdfPlumberAdapter 초기화 시작")
        config = Config.PDFPLUMBER

        self.pdf_path = Config.DOCUMENT_PATH
        self.output_dir = config["OUTPUT_IMAGE_DIR"]
        self.output_json = config["OUTPUT_JSON"]

        self.document_name = os.path.basename(self.pdf_path)
        LayoutDetectionLogger.info("PdfPlumberAdapter 초기화 완료")

    def parse(self):
        os.makedirs(self.output_dir, exist_ok=True)
        results = []

        with pdfplumber.open(self.pdf_path) as pdf:
            LayoutDetectionLogger.info(f"{len(pdf.pages)} 페이지 PDF 로드 완료.")

            for page_num, page in enumerate(pdf.pages):
                LayoutDetectionLogger.info(f"{page_num + 1} 페이지 분석 중...")

                words = page.extract_words()
                page_width = page.width
                page_height = page.height

                image_np = np.ones((int(page_height), int(page_width), 3), dtype=np.uint8) * 255

                for word in words:
                    x0, top, x1, bottom = word['x0'], word['top'], word['x1'], word['bottom']
                    text = word['text']

                    result = BaseDetectionResult(
                        document_name=self.document_name,
                        page_num=page_num + 1,
                        label="Text",
                        bbox=[int(x0), int(top), int(x1), int(bottom)],
                        score=None,
                        text=text.strip()
                    )
                    results.append(result)

                    cv2.rectangle(image_np, (int(x0), int(top)), (int(x1), int(bottom)), (255, 0, 0), 1)

                save_path = os.path.join(self.output_dir, f"page_{page_num + 1}.jpg")
                cv2.imwrite(save_path, image_np)
                LayoutDetectionLogger.info(f"시각화 이미지 저장: {save_path}")

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        LayoutDetectionLogger.info(f"전체 결과가 {self.output_json}에 저장되었습니다.")
