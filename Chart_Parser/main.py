import json
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Union
from PIL import Image
import openai
import argparse

def load_prompts(prompt_dir: Path) -> Dict[str, str]:
    """프롬프트 텍스트 파일(.txt)을 모두 읽어 딕셔너리로 반환"""
    prompts = {}
    for file in prompt_dir.glob("*.txt"):
        prompts[file.stem] = file.read_text(encoding="utf-8").strip()
    return prompts

def crop_and_encode(
    image_path: Path,
    bbox: List[int]
) -> str:
    """이미지 경로와 bbox 리스트(x1,y1,x2,y2)를 받아 크롭 후 Base64로 인코딩하여 반환"""
    x1, y1, x2, y2 = bbox
    with Image.open(image_path) as img:
        cropped = img.crop((x1, y1, x2, y2))
        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def evaluate(
    image_b64: str,
    prompt: str,
    api_key: str
) -> str:
    """GPT에 이미지(Base64)와 프롬프트를 함께 전송"""
    client = openai.OpenAI(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChartParser: layoutparser 결과 JSON에서 bbox를 받아 이미지 크롭 후 GPT 분석 요청"
    )
    parser.add_argument("--layout_json", type=Path, required=True,
                        help="layoutparser가 생성한 JSON 파일 경로 (bbox 리스트 포함)")
    parser.add_argument("--images_dir", type=Path, required=True,
                        help="layoutparser가 저장한 페이지 이미지 디렉토리")
    parser.add_argument("--prompts_dir", type=Path, required=True,
                        help="프롬프트 텍스트 파일(.txt) 디렉토리")
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API 키")
    args = parser.parse_args()

    # 1) layoutparser 결과 JSON 로드
    with open(args.layout_json, "r", encoding="utf-8") as f:
        detections: List[Dict] = json.load(f)
    print(f"🔍 Loaded {len(detections)} detections from {args.layout_json}")
    if not detections:
        print("⚠️ No detections found. layout_json 경로를 확인하세요.")
        exit(1)

    # 2) 프롬프트 로드
    prompts = load_prompts(args.prompts_dir)
    base_prompt = prompts.get("chart_analysis", "차트 내용을 분석해줘.")

    # 3) 디버그: 처음 5개 디텍션 확인
    for idx, det in enumerate(detections[:5]):
        print(f"➡️ det[{idx}]: page={det.get('page_num')}, label={det.get('label')}, bbox={det.get('bbox')}")

    # 4) 각 detection에 대해 이미지 크롭 및 GPT 호출
    results = []
    for det in detections:
        page = det.get("page_num")
        bbox = det.get("bbox", [])
        doc_name = det.get("document_name", f"page_{page}")

        img_file = args.images_dir / f"page_{page}.jpg"
        if not img_file.exists():
            print(f"❌ Missing image file: {img_file}")
            continue
        if not bbox or len(bbox) != 4:
            print(f"❌ Invalid bbox for det: {det}")
            continue

        # 이미지 크롭 및 GPT 호출
        image_b64 = crop_and_encode(img_file, bbox)
        prompt = base_prompt
        result = evaluate(image_b64, prompt, args.api_key)

        print(f"🎯 Page {page} {doc_name} 차트 분석 결과: {result}")

        results.append({
            "page": page,
            "document_name": doc_name,
            "bbox": bbox,
            "prompt_type": "chart_analysis",
            "result": result
        })

    # 5) 결과 저장
    output_path = Path("llm_a_outputs") / f"{doc_name}.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"✅ 결과 저장 완료: {output_path}")
