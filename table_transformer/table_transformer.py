from PIL import Image
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection

import torch
import matplotlib.pyplot as plt
import os
import psutil
import time
from transformers import DetrFeatureExtractor
import pandas as pd
from tabulate import tabulate

import pytesseract

from enum import Enum


class TableStructureExtractor:
    def __init__(self):
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.feature_extractor = DetrFeatureExtractor()

    def plot_results(self, pil_img, scores, labels, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            text = f'{self.model.config.id2label[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig('savefig_default.png') # 임시

    def plot_results_specific(self, pil_img, scores, labels, boxes, lab, ocr_data):
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
            if label == lab:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
                text = f'{self.model.config.id2label[label]}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
                
                break
        if ocr_data:
            for entry in ocr_data:
                xmin, ymin, xmax, ymax = entry['bbox']
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))

        plt.axis('off')
        plt.savefig('savefig_default.png') # 임시

    def draw_box_specific(self, image_path, labelnum, ocr_data=None):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)

        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.7)[0]
        self.plot_results_specific(image, results['scores'], results['labels'], results['boxes'], labelnum, ocr_data)
        print(results)
    
    def cell_detection(self, file_path):
        image = Image.open(file_path).convert("RGB")
        width, height = image.size

        encoding = self.feature_extractor(image, return_tensors="pt")
        encoding.keys()

        with torch.no_grad():
            outputs = self.model(**encoding)


        target_sizes = [image.size[::-1]]
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
        self.plot_results(image, results['scores'], results['labels'], results['boxes'])

    def compute_structure(self, image_path, threshold=0.7):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encoding)

        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(height, width)]
        )[0]
        return results['boxes'].tolist(), results['labels'].tolist()
    
    def extract_table(self, image_path, ocr_data, structure_threshold=0.7):
        """
        Build a DataFrame of the table by combining detected cell grid (structure) with OCR output.

        Uses IoU to assign each text entry to the cell with the highest overlap.
        """
        # 1. Detect table structure (rows and columns separators)
        boxes, labels = self.compute_structure(image_path, threshold=structure_threshold)

        # Separate row and column lines: assume label 2 = row, label 1 = column
        row_lines = sorted([b for b, l in zip(boxes, labels) if l == 2], key=lambda b: b[1])
        col_lines = sorted([b for b, l in zip(boxes, labels) if l == 1], key=lambda b: b[0])

        # 2. Build cell grid based on adjacent separators
        cell_boxes = []  # list of (bbox, row_idx, col_idx)
        for i in range(len(row_lines) - 1):
            for j in range(len(col_lines) - 1):
                ymin = row_lines[i][1]
                ymax = row_lines[i+1][3]
                xmin = col_lines[j][0]
                xmax = col_lines[j+1][2]
                cell_boxes.append(((xmin, ymin, xmax, ymax), i, j))

        # 3. Initialize empty table
        n_rows = len(row_lines) - 1
        n_cols = len(col_lines) - 1
        table = [["" for _ in range(n_cols)] for _ in range(n_rows)]

        # IoU helper
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            unionArea = boxAArea + boxBArea - interArea
            return interArea / unionArea if unionArea > 0 else 0

        # 4. Assign OCR entries to cells by highest IoU
        for entry in ocr_data:
            tbbox = entry['bbox']  # [xmin, ymin, xmax, ymax]
            best_iou = 0
            best_cell = None
            for (cbbox, r, c) in cell_boxes:
                overlap = iou(tbbox, cbbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_cell = (r, c)
            if best_cell is not None:
                r, c = best_cell
                text = entry['text'].strip()
                if table[r][c]:
                    table[r][c] += ' ' + text
                else:
                    table[r][c] = text

        # 5. Build pandas DataFrame and set headers from first row
        df = pd.DataFrame(table)
        headers = df.iloc[0].tolist()
        df = df[1:].copy()
        df.columns = headers
        return df
        
def generate_ocr_data(image_path):
    """
    Run pytesseract externally with --psm 11 to get OCR data.

    Returns:
        List[dict]: each dict contains 'text' and 'bbox' = [xmin, ymin, xmax, ymax]
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    # Use image_to_data for bounding boxes
    ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config='--psm 11')
    ocr_df = ocr_df.dropna(subset=['text'])

    ocr_data = []
    for _, row in ocr_df.iterrows():
        text = str(row['text']).strip()
        if not text:
            continue
        xmin, ymin, width, height = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
        bbox = [xmin, ymin, xmin + width, ymin + height]
        ocr_data.append({'text': text, 'bbox': bbox})
    return ocr_data

image_path = "sample_data/table.png"
extractor = TableStructureExtractor()
ocr_data = generate_ocr_data(image_path)
extractor.draw_box_specific(image_path, 2, ocr_data)
df = extractor.extract_table(image_path, ocr_data)
print(tabulate(df, headers="keys", tablefmt="psql"))