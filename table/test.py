from table_structure_extractor import TableStructureExtractor

import os
import yaml
from pathlib import Path

from io import BytesIO
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from bs4 import BeautifulSoup

def save_pairs_as_html(pairs, output_path='output.html',
                       img_max_width='200px', gap='1em'):
    """
    pairs: [(img_path, html_table_string), ...]
    """
    header = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        .pair {{ display: flex; align-items: flex-start; gap: {gap}; margin-bottom: 1em; }}
        .pair img {{ max-width: {img_max_width}; height: auto; border: 1px solid #ccc; padding: 4px; }}
        table, th, td {{
            border: 1px solid black;
            font-size: 10px;
        }}
      </style>
    </head>
    <body>
    """

    body = ""
    for img_path, html_str in pairs:
        body += f"""
        <div class="pair">
          <img src="{img_path}" />
          <div>{html_str}</div>
        </div>
        """

    footer = "</body></html>"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + body + footer)

    print("file_saved,", output_path)

def html_table_to_data_and_spans(html_str):
    """
    HTML <table>…</table> → 
      - data: 2D 리스트 (정확한 위치에 텍스트만)
      - spans: [(col_start, row_start, col_end, row_end), ...]
    """
    soup = BeautifulSoup(html_str, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    # 1) 전체 컬럼 수 계산 (각 row 의 colspan 합 중 최대값)
    n_cols = 0
    for row in rows:
        count = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['th','td']))
        n_cols = max(n_cols, count)

    # 2) data 초기화, occupancy 맵 준비
    data = [['' for _ in range(n_cols)] for _ in rows]
    occupancy = [[False]*n_cols for _ in rows]
    spans = []

    # 3) 셀 하나씩 돌면서
    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(['th','td']):
            # 이미 채워진 슬롯은 건너뛰기
            while c < n_cols and occupancy[r][c]:
                c += 1

            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(strip=True)

            # (r, c) 위치부터 colspan × rowspan 영역을 occupancy 처리
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupancy[r+dr][c+dc] = True

            # 정작 보여줄 텍스트는 (r,c) 에만
            data[r][c] = text

            # 병합 정보 기록
            if colspan > 1 or rowspan > 1:
                spans.append((c, r, c+colspan-1, r+rowspan-1))

            c += colspan

    return data, spans

def save_pairs_with_reportlab(pairs, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elems = []

    for img_path, html_str in pairs:
        # 1) 이미지
        pil = PILImage.open(img_path)
        bio = BytesIO()
        pil.save(bio, format='PNG', dpi=pil.info.get('dpi', (72,72)))
        bio.seek(0)

        elems.append(Image(bio))
        elems.append(Spacer(1, 12))

        # 2) HTML 테이블 → 단순화: <table> 내부만 읽어서 2D 리스트로 변환
        data, spans = html_table_to_data_and_spans(html_str)
        tbl = Table(data)
        # 기본 스타일 + SPAN 적용
        style = [
            ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ]
        for (c0, r0, c1, r1) in spans:
            style.append(('SPAN', (c0, r0), (c1, r1)))
        tbl.setStyle(TableStyle(style))

        elems.append(tbl)
        elems.append(Spacer(1, 24))

    doc.build(elems)

if __name__ == "__main__":
    config_path = Path("config_table_extractor.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    extractor = TableStructureExtractor(cfg["model"])

    image_code_pairs = []

    image_dir = "./sample_data"
    image_paths = os.listdir(image_dir)
    for i, image_path in enumerate(image_paths):
        image_full_path = os.path.join(image_dir, image_path)
        result_code = extractor.extract(image_full_path, None, f"./result/{i}.html", "html")
        image_code_pairs.append((image_full_path, result_code))

    result_path_html = "./result/result.html"
    result_path_pdf = "./result/result.pdf"
    save_pairs_as_html(image_code_pairs, result_path_html)
    save_pairs_with_reportlab(image_code_pairs, result_path_pdf)