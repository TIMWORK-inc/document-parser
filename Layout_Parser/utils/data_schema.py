from typing import List, Optional

class BaseDetectionResult:
    def __init__(
        self,
        document_name: str,
        page_num: int,
        label: str,
        bbox: List[int],
        score: Optional[float],
        text: str
    ):
        self.document_name = document_name
        self.page_num = page_num
        self.label = label
        self.bbox = bbox  # [x0, y0, x1, y1]
        self.score = score
        self.text = text

    def to_dict(self):
        return {
            "document_name": self.document_name,
            "page_num": self.page_num,
            "label": self.label,
            "bbox": self.bbox,
            "score": self.score,
            "text": self.text
        }
