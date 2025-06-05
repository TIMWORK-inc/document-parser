from calling_api.easy_ocr import EasyOCR
from calling_api.paddle_ocr import PaddleOCRWrapper
from calling_api.google_ocr import GoogleOCR
from calling_api.doctr_ocr import DocTROCR

from collections import defaultdict, Counter
from PIL import Image
import tempfile, os
import matplotlib.pyplot as plt

'''
log

1. metadata 완성

2. call 메서드 완성

3. cropped_img에 대해 각 모듈이 여러 개의 단어로 인식하거나 아무 단어도 인식 못할 때 <- 완료
-> 여러 단어일 땐 join
-> 아무 단어도 인식 못할 땐 무투표 처리

4. 왜 Ensembled된 결과로 공백이 나오지? -> 어떤 모듈도 단어를 인식 못함
-> base_ocr이 만든 text_box가 너무 작아서 생긴 문제 -> 고정 픽셀 패딩 적용 

고정 픽셀 패딩은 없애는게 낫다. 옆 글자까지 애매하게 잘리는 경우가 많아서
따라서 너무 작은 박스에 대해선 옆 박스랑 더하는게 나은듯



5. to do ->

* 문제 상황 발생_2
: 자른 이미지에 대해 sequential하게 매번 ocr 모듈을 호출하므로 너무 느리다.
모듈을 호출할 때 오버헤드가 상당히 큰 것 같다.
게다가 만약, google_ocr과 같이 유료 모듈을 사용한다면, 사용 요금이 너무 많아진다.

* 해결책
1. 자른 이미지에 대해 ocr 모듈을 여러 쓰레드로 병렬 처리한다.
- 그러나, 여전히 유료 ocr을 사용할 때 요금이 너무 많이 발생하는 문제를 해결하지 못한다.

2. 유료 API는 일부 샘플에만 적용
- low confidence 혹은 투표가 많이 갈리는 항목만 유료 OCR 재처리


'''


class OCR:
    def __init__(self, img_path, base_ocr, ocrs):
        """초기화 함수

        Args:
            img_path (str): 입력 이미지 경로
            base_ocr (object): 기준 OCR 엔진 인스턴스
            ocrs (List[object]): OCR 엔진 인스턴스 리스트
        """
        self.img_path = img_path
        self.base_ocr = base_ocr
        self.ocrs = ocrs

    def img_to_metadata(self):
        """기준 OCR에서 단어 단위 bbox 추출 후 정렬 및 인덱스 부여

        Returns:
            List[Dict[str, Any]]: index와 bbox를 포함한 메타데이터 리스트
        """
        results = self.base_ocr()
        texts = [item["text"] for item in results]
        print("base_ocr_result: ", texts)

        def get_center(bbox):
            x1, y1, x2, y2 = bbox
            return (round((y1 + y2) / 2, 1), round((x1 + x2) / 2, 1))

        results_sorted = sorted(results, key=lambda item: get_center(item['bbox']))
        metadata = [{"index": i, "bbox": item['bbox']} for i, item in enumerate(results_sorted)]
        return metadata

    @staticmethod
    def pad_bbox(bbox, pad=5):
        """고정된 픽셀 수만큼 bbox에 padding을 적용

        Args:
            bbox (List[int]): [x1, y1, x2, y2] 형태의 바운딩 박스
            pad (int, optional): padding 크기. Defaults to 5.

        Returns:
            List[int]: padding이 적용된 bbox
        """
        x1, y1, x2, y2 = bbox
        return [x1 - pad, y1 - pad, x2 + pad, y2 + pad]

    def calling_apis(self, cropped_img):
        """여러 OCR 엔진을 호출하여 결과 수집

        Args:
            cropped_img (str): 잘린 이미지 파일 경로

        Returns:
            List[List[Dict[str, Any]]]: 각 OCR 결과 리스트
        """
        results = []
        for ocr_engine in self.ocrs:
            try:
                ocr_engine.img = cropped_img
                result = ocr_engine()
                results.append(result)
            except Exception as e:
                print(f"failed: {e}")
                results.append([])
        return results

    @staticmethod
    def ensemble(results):
        """OCR 결과 앙상블 (다수결 + confidence 평균)

        Args:
            results (List[List[Dict[str, Any]]]): 각 OCR 결과 리스트

        Returns:
            List[Dict[str, Union[str, float]]]: 앙상블된 최종 결과
        """
        def join_result(words):
            if not words:
                return "", 0.0
            texts = [w["text"].strip() for w in words if w.get("text")]
            confs = [w["confidence"] for w in words if "confidence" in w]
            return " ".join(texts), sum(confs) / len(confs) if confs else 0.0

        texts, confs = [], []
        for result in results:
            text, conf = join_result(result)
            texts.append(text)
            confs.append(conf)

        votes = defaultdict(list)
        for text, conf in zip(texts, confs):
            if text.strip():
                votes[text].append(conf)

        if not votes:
            return [{"text": "", "confidence": 0.0}]

        vote_counts = Counter({t: len(c) for t, c in votes.items()})
        most_common = vote_counts.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            chosen_text = most_common[0][0]
        else:
            tied = [t for t, cnt in most_common if cnt == most_common[0][1]]
            chosen_text = max(tied, key=lambda t: sum(votes[t]) / len(votes[t]))

        avg_conf = sum(votes[chosen_text]) / len(votes[chosen_text])
        return [{"text": chosen_text, "confidence": avg_conf}]

    def __call__(self):
        """OCR 처리 파이프라인 실행

        이미지에서 텍스트 박스를 추출하고 각 박스에 대해 여러 OCR을 적용 후 결과 앙상블 수행
        """
        def save_temp_image(pil_img):
            """PIL 이미지를 임시 파일로 저장하고 경로 반환

            Args:
                pil_img (PIL.Image): 이미지 객체

            Returns:
                str: 임시 저장 경로
            """
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(temp_file.name)
            return temp_file.name

        metadata = self.img_to_metadata()
        img = Image.open(self.img_path)

        for item in metadata:
            padded_bbox = self.pad_bbox(item["bbox"], 0)
            cropped_img = img.crop(padded_bbox)
            
            # 이미지 잘 잘리는가 확인
            script_dir = os.path.dirname(os.path.abspath(__file__))  # 항상 파일 기준
            output_dir = os.path.join(script_dir, "test_imgs")
            cropped_img.save(os.path.join(output_dir, f"{item['index']}.png"))

            
            temp_path = save_temp_image(cropped_img)

            try:
                results = self.calling_apis(temp_path)
                r_ensembled = self.ensemble(results)
                print(f"index: {item['index']}, result: {r_ensembled}")
            finally:
                os.remove(temp_path)

def main():
    """OCR 실행 예제"""
    path = "./docs/dummy_1.png"
    base_ocr = EasyOCR(path)
    ocrs = [EasyOCR, PaddleOCRWrapper, DocTROCR]
    ocr = OCR(path, base_ocr=base_ocr, ocrs=[ocr(path) for ocr in ocrs])
    ocr()

if __name__ == '__main__':
    main()
