from PIL import Image
import torch
from torch import nn, Tensor
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from bs4 import BeautifulSoup as bs
import json

import pytesseract

from IPython.display import display, HTML

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import re
import tokenizers as tk

from functools import partial
import warnings

from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import (
    subsequent_mask,
    pred_token_within_range,
    greedy_sampling,
    bbox_str_to_token_list,
    cell_str_to_token_list,
    html_str_to_token_list,
    build_table_from_html_and_cell,
    html_table_template,
)
from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN

warnings.filterwarnings('ignore')
device = torch.device("cuda:0")

MODEL_FILE_NAME = [
    "unitable_large_structure.pt",  # Structure model weights
    "unitable_large_bbox.pt",       # Bounding box model weights
    "unitable_large_content.pt",    # Content model weights
]
MODEL_DIR = Path("./experiments/unitable_weights")
assert all([(MODEL_DIR / name).is_file() for name in MODEL_FILE_NAME]), \
    "Please download model weights from HuggingFace: https://huggingface.co/poloclub/UniTable/tree/main"


class TableStructureExtractor:
    """
    Extracts table structure from an image using a pretrained UniTable model.
    """

    def __init__(self, vocab_path: str = "./vocab/vocab_html.json", max_seq_len: int = 784, model_weights: Union[str, Path] = MODEL_DIR / MODEL_FILE_NAME[0]) -> None:
        """
        TableStructureExtractor를 초기화합니다.

        Args:
            vocab_path (str): HTML 어휘가 저장된 JSON 파일의 경로
            max_seq_len (int): 디코딩할 최대 시퀀스 길이
            model_weights (str | Path): 사전 학습된 모델 가중치 파일의 경로
        """
        d_model = 768
        patch_size = 16
        nhead = 12
        dropout = 0.2

        # 모델 구성 요소 초기화
        self.backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
        self.encoder = Encoder(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            nlayer=12,
            ff_ratio=4,
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            nlayer=4,
            ff_ratio=4,
        )

        # 토크나이저 및 토큰 설정
        self.vocab = tk.Tokenizer.from_file(vocab_path)
        self.prefix = [self.vocab.token_to_id("[html]")]
        self.eos_id = self.vocab.token_to_id("<eos>")
        self.token_whitelist = [self.vocab.token_to_id(i) for i in VALID_HTML_TOKEN]

        # 인코더-디코더 모델 빌드
        self.model = EncoderDecoder(
            backbone=self.backbone,
            encoder=self.encoder,
            decoder=self.decoder,
            vocab_size=self.vocab.get_vocab_size(),
            d_model=d_model,
            padding_idx=self.vocab.token_to_id("<pad>"),
            max_seq_len=max_seq_len,
            dropout=dropout,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # 사전 학습된 가중치 로드
        self.model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        self.model = self.model.to(device)

    def extract(
        self,
        image_path: str,
        ocr_data: Optional[List[Dict[str, Any]]],
        save_path: str = "./result.html",
        max_decode_len: int = 512,
        output_format: str = "html",
    ) -> Union[str, Dict[str, Any]]:
        """이미지에서 테이블 구조를 추출합니다.

        Args:
            image_path (str): 테이블이 포함된 입력 이미지 파일 경로
            ocr_data (List[dict] | None): OCR 결과 데이터(텍스트와 바운딩 박스 목록). None인 경우 placeholder를 사용
            save_path (str): 결과 파일을 저장할 경로
            max_decode_len (int): 디코딩을 수행할 최대 토큰 수
            output_format (str): "html" 또는 "json" 중 선택

        Returns:
            Union[str, dict]: HTML 문자열 또는 JSON 객체
        """
        image = Image.open(image_path).convert("RGB")

        # 이미지 전처리 및 텐서 변환
        T = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.86597056, 0.88463002, 0.87491087],
                std=[0.20686628, 0.18201602, 0.18485524],
            ),
        ])
        image_tensor = T(image)
        image_tensor = image_tensor.to(device).unsqueeze(0)

        # 모델을 평가 모드로 설정하고 인코딩
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(image_tensor)
            context = torch.tensor(self.prefix, dtype=torch.int32).repeat(image_tensor.shape[0], 1).to(device)

        # 디코딩을 통해 HTML 토큰 생성
        for _ in range(max_decode_len):
            eos_flag = [self.eos_id in k for k in context]
            if all(eos_flag):
                break
            with torch.no_grad():
                causal_mask = subsequent_mask(context.shape[1]).to(device)
                logits = self.model.decode(
                    memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
                )
                logits = self.model.generator(logits)[:, -1, :]
            logits = pred_token_within_range(
                logits.detach(),
                white_list=self.token_whitelist,
                black_list=None,
            )
            next_probs, next_tokens = greedy_sampling(logits)
            context = torch.cat([context, next_tokens], dim=1)

        pred_html_ids = context.detach().cpu().numpy()[0]
        pred_html_str = self.vocab.decode(pred_html_ids, skip_special_tokens=False)
        pred_html_tokens = html_str_to_token_list(pred_html_str)

        html_code = []
        if ocr_data is None:
            ocr_data = [{'text': "placeholder"}] * len(pred_html_tokens)

        # OCR 데이터를 이용해 <td> 태그를 실제 텍스트로 교체
        for tag in pred_html_tokens:
            if tag in ("<td>[]</td>", ">[]</td>"):
                if len(ocr_data) == 0:
                    continue
                cell = ocr_data.pop(0)
                cell_text = cell['text']
                html_code.append(tag.replace("[]", cell_text))
            else:
                html_code.append(tag)
        html_code_str = "".join(html_code)
        html_code_str = html_table_template(html_code_str)

        soup = bs(html_code_str)
        table_code = soup.prettify()

        if output_format == "json":
            table_json = self._html_to_json(table_code)
            with open(save_path, "w") as f:
                json.dump(table_json, f, ensure_ascii=False, indent=2)
            return table_json
        else:
            with open(save_path, "w") as f:
                f.write(table_code)
            return table_code

    def _html_to_json(self, html: str) -> Dict[str, Any]:
        """Convert HTML table code to a simple Textract-like JSON."""
        soup = bs(html, "html.parser")
        table = soup.find("table")
        cells = []
        row_idx = 1
        for tr in table.find_all("tr"):
            col_idx = 1
            for td in tr.find_all("td"):
                row_span = int(td.get("rowspan", 1))
                col_span = int(td.get("colspan", 1))
                text = td.get_text(strip=True)
                cells.append(
                    {
                        "RowIndex": row_idx,
                        "ColumnIndex": col_idx,
                        "RowSpan": row_span,
                        "ColumnSpan": col_span,
                        "Text": text,
                    }
                )
                col_idx += col_span
            row_idx += 1
        return {"Cells": cells}


def generate_ocr_data(image_path: str) -> List[Dict[str, Any]]:
    """
    이미지를 입력받아 Tesseract OCR을 통해 텍스트와 바운딩 박스를 추출합니다.

    Args:
        image_path (str): OCR을 수행할 이미지 파일 경로

    Returns:
        List[dict]: 텍스트와 bbox 정보를 담은 딕셔너리 목록
    """
    image = Image.open(image_path).convert("RGB")
    ocr_df = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DATAFRAME,
        config='--psm 11'
    )
    ocr_df = ocr_df.dropna(subset=['text'])

    ocr_data = []
    for _, row in ocr_df.iterrows():
        text = str(row['text']).strip()
        if not text:
            continue
        xmin = int(row['left'])
        ymin = int(row['top'])
        width = int(row['width'])
        height = int(row['height'])
        bbox = [xmin, ymin, xmin + width, ymin + height]
        ocr_data.append({'text': text, 'bbox': bbox})
    return ocr_data

if __name__ == "__main__":
    image_path = "sample_data/PMC2838834_005_00.png"

    extractor = TableStructureExtractor()

    ocr_data = generate_ocr_data(image_path)

    pred_html = extractor.extract(
        image_path, ocr_data, save_path="./result_with_ocr.html", output_format="html"
    )
    pred_json = extractor.extract(
        image_path, ocr_data, save_path="./result.json", output_format="json"
    )

    pred_html = extractor.extract(
        image_path, None, save_path="./result_without_ocr.html", output_format="html"
    )

    print("Extraction complete. HTML saved.")
