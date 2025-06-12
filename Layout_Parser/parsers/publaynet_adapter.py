import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract

from config.config import Config
from utils.data_schema import BaseDetectionResult

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# 새 Logger 시스템 import
from Logger import LayoutDetectionLogger

class PubLayNetAdapter:
    def __init__(self):
        LayoutDetectionLogger.info("PubLayNetAdapter 초기화 시작")
        config = Config.PUBLAYNET

        self.pdf_path = Config.DOCUMENT_PATH
        self.model_path = config["MODEL_PATH"]
        self.score_thresh = config["SCORE_THRESH"]
        self.label_map = config["LABEL_MAP"]
        self.num_classes = config["NUM_CLASSES"]

        self.output_dir = config["OUTPUT_IMAGE_DIR"]
        self.output_json = config["OUTPUT_JSON"]

        self.model = self._load_model()

        self.document_name = os.path.basename(self.pdf_path)
        LayoutDetectionLogger.info("PubLayNetAdapter 초기화 완료")

    def _load_model(self):
        LayoutDetectionLogger.info("Detectron2 모델 로드 시작")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        LayoutDetectionLogger.info("Detectron2 모델 로드 완료")
        return DefaultPredictor(cfg)

    def parse(self):
        os.makedirs(self.output_dir, exist_ok=True)

        images = convert_from_path(self.pdf_path, dpi=Config.DPI)
        LayoutDetectionLogger.info(f"{len(images)} 페이지의 PDF가 이미지로 변환되었습니다.")

        results = []

        for page_num, img_pil in enumerate(images):
            LayoutDetectionLogger.info(f"{page_num + 1} 페이지 분석 중...")
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            outputs = self.model(image)

            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()
            classes = outputs["instances"].pred_classes.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                label = self.label_map[int(cls)]
                segment_img = image[y1:y2, x1:x2]
                text = pytesseract.image_to_string(segment_img, lang='kor+eng')

                result = BaseDetectionResult(
                    document_name=self.document_name,
                    page_num=page_num + 1,
                    label=label,
                    bbox=[x1, y1, x2, y2],
                    score=float(score),
                    text=text.strip()
                )
                results.append(result)

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box.astype(int)
                label = self.label_map[int(cls)]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_path = os.path.join(self.output_dir, f"page_{page_num + 1}.jpg")
            cv2.imwrite(save_path, image)
            LayoutDetectionLogger.info(f"시각화 이미지 저장: {save_path}")

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        LayoutDetectionLogger.info(f"전체 결과가 {self.output_json}에 저장되었습니다.")
