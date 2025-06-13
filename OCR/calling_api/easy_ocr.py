# 라이브러리 임포트
import easyocr
import re
class EasyOCR():
    '''
    EasyOCR의 output은 원래 hierarchy가 아님.
    text, bbox, conf만 List[Dict]으로 반환하는 클래스
    '''
    def __init__(self, img):
        """
        EasyOCR 객체 초기화

        Args: img (str): 이미지 경로
        """
        self.img = img
        
    def processing(self):
        '''
        OCR 엔진 호출
        
        Returns: List[Tuple]: EasyOCR 결과 [(bbox, text, confidence), ...]
        '''

        reader = easyocr.Reader(['ko', 'en'])
        results = reader.readtext(self.img)
        
        return results

    @staticmethod
    def post_processing(raw_result):
        """
        후처리: bounding box, text, confidence 포맷 통일 + 단어 단위 재청킹

        Args:
            raw_result (List): EasyOCR 결과 [(bbox, text, confidence), ...]

        Returns:
            List[Dict]: 통일된 포맷의 결과 (단어 단위로 재청킹됨)
        """
        formatted = []
        # 빈 결과 처리
        if not raw_result:
            return formatted

        for (bbox, text, confidence) in raw_result:
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            width = x2 - x1

            # 단어 및 구두점 단위로 토큰화
            tokens = re.findall(r'\w+|[^\w\s]', text)
            if not tokens:
                continue

            # 토큰 개수로 bbox를 비례 분할
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
                    "engine": "EasyOCR",
                    "hierarchy": None
                })

        return formatted
        
    def __call__(self):
        """
        전체 OCR 파이프라인 실행
        
        Returns: List[Dict]: 최종 포맷 결과
        """
        raw = self.processing()
        output = self.post_processing(raw)
        
        return output
    
    
# ✅ 메인 함수에서 테스트
if __name__ == "__main__":
    image_path = "./docs/dummy_1.png"  # 테스트할 이미지 경로
    ocr = EasyOCR(image_path)
    results = ocr()

    print("=== OCR 결과 ===")
    for item in results:
        print(f"[Text]: {item['text']}")
        print(f" BBox : {item['bbox']}")
        print(f" Conf : {item['confidence']:.2f}")
        print()
            
