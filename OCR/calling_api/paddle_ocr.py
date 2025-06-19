# 라이브러리 임포트
from paddleocr import PaddleOCR as PaddleOCRLib
import re

class PaddleOCRWrapper:
    """
    PaddleOCR의 output은 원래 hierarchy가 아님.
    text, bbox, conf만 투플로 묶은 뒤 리스트로 출력하는 클래스
    """
    def __init__(self, img):
        """
        PaddleOCRWrapper 객체 초기화

        Args:
            img (str): 이미지 경로
        """
        self.img = img
        self.reader = PaddleOCRLib(use_angle_cls=True, lang='en', show_log=False)

    def processing(self):
        """
        OCR 엔진 호출

        Returns:
            List[Tuple]: PaddleOCR 결과 [(bbox, (text, confidence)), ...]
        """
        results = self.reader.ocr(self.img, cls=True)
        return results[0]  # 첫 번째 이미지 결과만 사용
    

    @staticmethod
    def post_processing(raw_result):
        """
        후처리: bounding box, text, confidence 포맷 통일 + 단어 단위 재청킹

        Args:
            raw_result (List): PaddleOCR 결과 [(bbox, (text, confidence)), ...]

        Returns:
            List[Dict]: 통일된 포맷의 결과 (단어 단위로 재청킹됨)
        """
        formatted = []

        # 빈 결과 처리
        if not raw_result:
            return formatted
        
        for bbox, (text, confidence) in raw_result:
            x1, y1 = bbox[0]  # top-left
            x2, y2 = bbox[2]  # bottom-right
            width = x2 - x1

            tokens = re.findall(r'\w+|[^\w\s]', text)
            if not tokens:
                continue

            total_len = sum(len(token) for token in tokens)
            if total_len == 0:
                continue

            offset = 0
            for token in tokens:
                token_len = len(token)
                ratio = token_len / total_len
                token_width = width * ratio

                token_x1 = x1 + offset
                token_x2 = token_x1 + token_width
                offset += token_width

                formatted.append({
                    "text": token,
                    "bbox": [int(token_x1), int(y1), int(token_x2), int(y2)],
                    "confidence": confidence,
                    "engine": "PaddleOCR",
                    "hierarchy": None
                })

        return formatted


    def __call__(self):
        """
        전체 OCR 파이프라인 실행

        Returns:
            List[Dict]: 최종 포맷 결과
        """
        raw = self.processing()
        output = self.post_processing(raw)
        return output

