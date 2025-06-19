import re
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class DocTROCR:
    """
    doctr 기반 OCR 클래스

    이미지를 입력 받아 텍스트 인식 결과를 추출하고,
    각 단어를 일정한 포맷으로 정리하여 반환한다.
    """

    def __init__(self, img):
        """
        DocTROCR 객체 초기화

        Args:
            img (str): 이미지 파일 경로
        """
        self.img = img

    def processing(self):
        """
        doctr OCR 엔진 호출

        Returns:
            Dict: doctr의 raw OCR 예측 결과 (JSON 형식)
        """
        doc = DocumentFile.from_images(self.img)
        reader = ocr_predictor(pretrained=True)
        results = reader(doc).export()
        return results

    @staticmethod
    def post_processing(raw_result):
        """
        후처리: doctr 결과를 단어 단위로 분할하고 포맷을 통일한다.

        Args:
            raw_result (Dict): doctr의 JSON 형식 OCR 결과

        Returns:
            List[Dict]: {"text", "bbox", "confidence", "engine", "hierarchy"} 구조의 리스트
        """
        formatted = []
        # 빈 결과 처리
        if not raw_result:
            return formatted

        for page in raw_result['pages']:
            page_idx = page.get('page_idx', 0)
            for block in page['blocks']:
                block_box = block['geometry']
                for line in block['lines']:
                    line_box = line['geometry']
                    for word in line['words']:
                        word_box = word['geometry']
                        text = word['value']
                        confidence = word['confidence']

                        x1, y1 = word_box[0]
                        x2, y2 = word_box[1]
                        width = x2 - x1

                        # 단어 및 구두점 단위로 토큰화
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
                                "bbox": [round(token_x1), round(y1), round(token_x2), round(y2)],
                                "confidence": confidence,
                                "engine": "DocTR",
                                "hierarchy": {
                                    "page_idx": page_idx,
                                    "block_box": block_box,
                                    "line_box": line_box,
                                }
                            })

        return formatted

    def __call__(self):
        """
        전체 OCR 파이프라인 실행

        Returns:
            List[Dict]: 후처리된 일관된 포맷의 OCR 결과
        """
        raw = self.processing()
        return self.post_processing(raw)

