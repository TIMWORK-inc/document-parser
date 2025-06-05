import requests
import base64
import json
from dotenv import load_dotenv
import os
import re

class GoogleOCR:
    """
    Google Cloud Vision API를 사용한 OCR 클래스.
    
    이미지에서 텍스트를 인식하고, 결과를 단어 단위로 파싱하여
    통일된 포맷으로 반환한다.
    """

    def __init__(self, img):
        """
        GoogleOCR 객체 초기화

        Args:
            img (str): 이미지 파일 경로
        """
        self.img = img

    def processing(self):
        """
        Google Cloud Vision API 요청 수행

        Returns:
            dict: Vision API 원시 응답
        """
        load_dotenv()
        API_KEY = os.getenv("GOOGLE_API_KEY")
        if not API_KEY:
            raise ValueError("Google API Key가 .env에 설정되어 있지 않습니다.")

        with open(self.img, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode()

        url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        body = {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "TEXT_DETECTION"}]
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(body))
        response.raise_for_status()
        return response.json()

    @staticmethod
    def extract_box(vertices):
        """
        4개의 꼭짓점 좌표에서 [x1, y1, x2, y2] 형태로 변환

        Args:
            vertices (List[Dict]): {"x": ..., "y": ...} 꼭짓점 리스트

        Returns:
            List[int]: [x1, y1, x2, y2]
        """
        if len(vertices) < 4:
            return [0, 0, 0, 0]
        x1 = vertices[0].get('x', 0)
        y1 = vertices[0].get('y', 0)
        x2 = vertices[2].get('x', 0)
        y2 = vertices[2].get('y', 0)
        return [x1, y1, x2, y2]

    @classmethod
    def post_processing(cls, raw_result):
        """
        응답 결과 후처리: 단어 단위로 재청킹하고 포맷 통일

        Args:
            raw_result (dict): Google OCR API 응답

        Returns:
            List[Dict]: {"text", "bbox", "confidence", "engine", "hierarchy"}
        """
        formatted = []
        
        # 빈 결과 처리
        if not raw_result:
            return formatted
        
        try:
            pages = raw_result['responses'][0]['fullTextAnnotation']['pages']
        except (KeyError, IndexError):
            return formatted

        for page_idx, page in enumerate(pages):
            for block in page.get('blocks', []):
                block_box = cls.extract_box(block.get('boundingBox', {}).get('vertices', []))
                for para in block.get('paragraphs', []):
                    paragraph_box = cls.extract_box(para.get('boundingBox', {}).get('vertices', []))
                    for word in para.get('words', []):
                        word_box = cls.extract_box(word.get('boundingBox', {}).get('vertices', []))
                        text = ''.join(s.get('text', '') for s in word.get('symbols', []))
                        confidence = word.get('confidence', 1.0)

                        formatted.append({
                            "text": text,
                            "bbox": word_box,
                            "confidence": confidence,
                            "engine": "GoogleOCR",
                            "hierarchy": {
                                "page_idx": page_idx,
                                "block_box": block_box,
                                "paragraph_box": paragraph_box
                            }
                        })

        return formatted

    def __call__(self):
        """
        전체 OCR 파이프라인 실행

        Returns:
            List[Dict]: 최종 통일된 OCR 결과
        """
        raw = self.processing()
        return self.post_processing(raw)


def main():
    image_path = "./docs/dummy_1.png"
    ocr = GoogleOCR(image_path)
    try:
        results = ocr()
    except Exception as e:
        print(f"[오류 발생] {e}")
        return

    print("=== GoogleOCR 결과 ===")
    for item in results:
        print(f"Text      : {item['text']}")
        print(f"BBox      : {item['bbox']}")
        print(f"Confidence: {item['confidence']:.2f}")
        print(f"Engine    : {item['engine']}")
        print(f"Hierarchy : {item['hierarchy']}")
        print("---")

if __name__ == "__main__":
    main()
