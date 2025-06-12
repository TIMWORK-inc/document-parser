import layoutparser as lp
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import json
import os
from PIL import Image

from config.config import Config
from utils.data_schema import BaseDetectionResult

# Logger 시스템 import
from Logger import LayoutDetectionLogger

class LayoutParserAdapter:
    def __init__(self):
        LayoutDetectionLogger.info("LayoutParserAdapter 초기화 시작")

        config = Config.LAYOUT_PARSER

        self.pdf_path = Config.DOCUMENT_PATH
        self.model_path = config["MODEL_PATH"]
        self.score_thresh = config["SCORE_THRESH"]
        self.label_map = config["LABEL_MAP"]

        self.output_dir = config["OUTPUT_IMAGE_DIR"]
        self.output_json = config["OUTPUT_JSON"]

        self.model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            model_path=self.model_path,
            label_map=self.label_map,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.score_thresh]
        )

        self.document_name = os.path.basename(self.pdf_path)
        LayoutDetectionLogger.info("LayoutParserAdapter 초기화 완료")

    def parse(self):
        os.makedirs(self.output_dir, exist_ok=True)

        images = convert_from_path(self.pdf_path, dpi=Config.DPI)
        LayoutDetectionLogger.info(f"{len(images)} 페이지의 PDF가 이미지로 변환되었습니다.")

        results = []

        for page_num, img_pil in enumerate(images):
            LayoutDetectionLogger.info(f"{page_num + 1} 페이지 분석 중...")

            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            layout = self.model.detect(image)

            for block in layout:
                x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
                segment_img = block.crop_image(image)
                text = pytesseract.image_to_string(segment_img, lang='kor+eng')

                result = BaseDetectionResult(
                    document_name=self.document_name,
                    page_num=page_num + 1,
                    label=block.type,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    score=float(block.score),
                    text=text.strip()
                )
                results.append(result)

            visualized = lp.draw_box(image, layout, box_width=3)
            visualized_np = cv2.cvtColor(np.array(visualized), cv2.COLOR_RGB2BGR)

            save_path = os.path.join(self.output_dir, f"page_{page_num + 1}.jpg")
            cv2.imwrite(save_path, visualized_np)
            LayoutDetectionLogger.info(f"시각화 이미지 저장: {save_path}")

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        LayoutDetectionLogger.info(f"전체 결과가 {self.output_json}에 저장되었습니다.")
