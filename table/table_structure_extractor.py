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

import yaml

warnings.filterwarnings('ignore')
device = torch.device("cuda:0")


class TableStructureExtractor:
    """
    Extracts table structure from an image using a pretrained UniTable model.
    """

    def __init__(self, model_cfg: Dict) -> None:
        """
        TableStructureExtractor를 초기화합니다.

        Args:
            model_cfg (Dict): 모델의 변수가 저장된 딕셔너리(from config_table_extractor.yml/model)
        """
        vocab_path = model_cfg["vocab_path"]

        model_dir = Path(model_cfg["model_dir"])
        model_weights = model_dir / model_cfg["model_files"]["structure"]

        assert model_weights.is_file(), \
            "Please download model weights from HuggingFace: https://huggingface.co/poloclub/UniTable/tree/main"

        d_model = model_cfg["d_model"]
        patch_size = model_cfg["patch_size"]
        nhead = model_cfg["nhead"]
        dropout = model_cfg["dropout"]

        max_seq_len = model_cfg["max_seq_len"]

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
        save_path: str,
        output_format: str,
        max_decode_len: int = 528,
    ) -> str:
        """이미지에서 테이블 구조를 추출합니다.

        Args:
            image_path (str): 테이블이 포함된 입력 이미지 파일 경로
            ocr_data (List[dict] | None): OCR 결과 데이터(텍스트와 바운딩 박스 목록). None인 경우 placeholder(text로 표기)를 사용
            save_path (str): 결과 파일을 저장할 경로
            output_format (str): "html" 또는 "json" 중 선택
            max_decode_len (int): 디코딩을 수행할 최대 토큰 수

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
            ocr_data = [{'text': "text"}] * len(pred_html_tokens)

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
        table_html = html_table_template(html_code_str)

        result = None

        if output_format == "html":
            result = table_html
        elif output_format == "json":
            table_json = self._html_to_json(table_html)
            result = json.dumps(table_json)
        else:
            assert True, "output_format must be html or json!"

        if save_path:
            with open(save_path, "w") as f:
                f.write(result)

            print(f"Save complete. [{image_path}] -> [{save_path}]")
        else:
            print(f"Extraction complete. [{image_path}]")

        return result
            

    def _html_to_json(self, html: str) -> Dict[str, Any]:
        """HTML 테이블 코드를 간단한 Textract 유사 JSON으로 변환합니다.

        이 메서드는 주어진 HTML 문자열을 파싱하여 첫 번째 <table> 요소를 찾고,
        각 셀의 행/열 인덱스, span 정보, 텍스트 내용을 추출하여 AWS Textract와 유사한
        형식의 딕셔너리 리스트로 반환합니다.

        Args:
            html (str): 하나 이상의 <table> 요소를 포함하는 HTML 문자열.

        Returns:
            Dict[str, Any]: 키 "Cells"를 가지는 딕셔너리. 값은 각 테이블 셀을 나타내는 딕셔너리의 리스트이며,
                각 셀 딕셔너리는 다음 키를 포함합니다:
                - "RowIndex" (int): 1부터 시작하는 셀의 행 인덱스.
                - "ColumnIndex" (int): 1부터 시작하는 셀의 열 인덱스(ColSpan을 반영).
                - "RowSpan" (int): 셀이 차지하는 행 수.
                - "ColumnSpan" (int): 셀이 차지하는 열 수.
                - "Text" (str): 셀 내부의 텍스트 내용(양쪽 공백 제거).

        Raises:
            ValueError: 제공된 HTML에 <table> 요소가 없을 경우 발생합니다.
        """

        soup = bs(html, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError("제공된 HTML에 <table> 요소가 없습니다.")
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
    config_path = Path("config_table_extractor.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_path = "sample_data/0.png"

    extractor = TableStructureExtractor(cfg["model"])

    ocr_data = generate_ocr_data(image_path)

    pred_json = extractor.extract(
        image_path, None, save_path="./result/without_ocr.json", output_format="json"
    )

    pred_html = extractor.extract(
        image_path, None, save_path=None, output_format="html"
    )

    print(pred_html)
