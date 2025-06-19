import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rxconfig import GOOGLE_OCR_API_KEY

from calling_api.easy_ocr import EasyOCR
from calling_api.paddle_ocr import PaddleOCRWrapper
from calling_api.google_ocr import GoogleOCR
from calling_api.doctr_ocr import DocTROCR

from collections import defaultdict, Counter
import json, os, argparse
from ensemble_boxes import weighted_boxes_fusion

class OCR:
    def __init__(self, img_path, ocrs, ensemble = True):
        """
        OCR 클래스 초기화.

        Args:
            img_path (str): OCR을 수행할 이미지 경로.
            ocrs (List[object]): OCR 엔진 인스턴스 리스트.
            ensemble (bool): 앙상블 여부.
        """
        self.img_path = img_path
        self.ocrs = ocrs
        self.ensemble = ensemble

    def calling_apis(self):
        """
        [OCR_001] 전체 이미지 OCR 처리

        Returns:
            Dict[str, List[Dict[str, Any]]]: 각 OCR 엔진 이름별로 OCR 결과를 리스트 형태로 반환.
        """
        results_by_engine = {}
        for ocr_engine in self.ocrs:
            try:
                ocr_engine.img = self.img_path
                result = ocr_engine()
                results_by_engine[ocr_engine.__class__.__name__] = result
            except Exception as e:
                print(f"Failed {ocr_engine.__class__.__name__}: {e}")
                results_by_engine[ocr_engine.__class__.__name__] = []
        return results_by_engine

    def apply_wbf(self, results_by_engine):
        """
        [OCR_002] Weighted Boxes Fusion을 이용한 bbox 병합

        Args:
            results_by_engine (Dict[str, List[Dict[str, Any]]]): 각 OCR 엔진의 결과.

        Returns:
            List[Dict[str, Union[List[int], float]]]: 병합된 bbox와 평균 confidence 리스트.
        """
        boxes, scores, labels = [], [], []

        for engine, result in results_by_engine.items():
            b, s, l = [], [], []
            for r in result:
                x1, y1, x2, y2 = r["bbox"]
                b.append([x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]) # bbox를 0~1 사이 값으로 정규화 (WBF는 정규화된 좌표를 요구함)
                s.append(r.get("confidence", 1.0))  # confidence 점수
                l.append(0)   # class label (dummy)
            boxes.append(b)
            scores.append(s)
            labels.append(l)

        if not any(boxes):
            return []

        merged_boxes, merged_scores, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        # iou_thr=0.5: IoU가 50% 이상이면 같은 그룹으로 간주
        
        merged_results = [] # 병합된 박스를 다시 복원
        for box, score in zip(merged_boxes, merged_scores):
            x1, y1, x2, y2 = [int(c * 1000) for c in box]
            merged_results.append({"bbox": [x1, y1, x2, y2], "confidence": score})

        return merged_results

    def ensemble_text(self, merged_bboxes, results_by_engine):
        """
        [OCR_003] 텍스트 다수결 앙상블 수행

        Args:
            merged_bboxes (List[Dict]): 병합된 bbox 리스트.
            results_by_engine (Dict[str, List[Dict]]): 각 엔진별 원시 OCR 결과.

        Returns:
            List[Dict[str, Union[str, float, List[int]]]]: 최종 텍스트와 confidence 결과.
        """
        def iou(b1, b2):
            xa, ya, xb, yb = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0

        final_results = []
        for box in merged_bboxes:
            texts = defaultdict(list)
            for engine_results in results_by_engine.values():
                for r in engine_results:
                    if iou(r["bbox"], box["bbox"]) > 0.5:
                        text = r["text"].strip()
                        if text:
                            texts[text].append(r.get("confidence", 1.0))

            if not texts:
                final_results.append({"bbox": box["bbox"], "text": "", "confidence": 0.0})
                continue

            vote_counts = Counter({t: len(c) for t, c in texts.items()})
            most_common = vote_counts.most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                chosen_text = most_common[0][0]
            else:
                tied = [t for t, cnt in most_common if cnt == most_common[0][1]]
                chosen_text = max(tied, key=lambda t: sum(texts[t]) / len(texts[t]))

            avg_conf = sum(texts[chosen_text]) / len(texts[chosen_text])
            final_results.append({"bbox": box["bbox"], "text": chosen_text, "confidence": avg_conf})

        return final_results


    def __call__(self):
        """
        OCR 파이프라인 실행 메서드.

        수행 단계:
            1. 전체 이미지에 대해 OCR 호출
            2. WBF 병합
            3. 텍스트 앙상블
            4. 결과 출력 또는 저장
        """
        print("▶ 단계 1: OCR 엔진 호출")
        raw_results = self.calling_apis()
        if not self.ensemble:
            print("▶ 단일 엔진 결과만 출력 (앙상블 없음)")
            print(raw_results)

        print("▶ 단계 2: WBF로 BBox 병합")
        merged_bboxes = self.apply_wbf(raw_results)

        print("▶ 단계 3: 텍스트 앙상블")
        final_results = self.ensemble_text(merged_bboxes, raw_results)
        print(final_results)

def main():
    """
    OCR 파이프라인 예제 실행 함수.
    """
    
    # 1. 이미지 경로 처리 -----------------
    img_path = "../sample_docs/dummy_1.png"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, img_path)
    
    # 2. 설정 ----------------------------
    use_google = False  # Google OCR 사용할지 여부
    ensemble = True    # 앙상블할지 여부
    
    # 3. 사용할 OCR 엔진 설정 --------------
    ocrs = [
    EasyOCR(path),
    PaddleOCRWrapper(path),
    DocTROCR(path)
    ]
    if use_google:
        ocrs.append(GoogleOCR(path, GOOGLE_OCR_API_KEY))  # GoogleOCR은 callable로 래핑
    
    
    # 4. OCR 파이프라인 실행 --------------
    pipeline = OCR(path, ocrs, ensemble)
    pipeline()

if __name__ == "__main__":
    main()
