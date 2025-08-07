import base64
import openai
import json
import time
from pathlib import Path
from typing import Dict


def load_prompts(prompt_dir: Path) -> Dict[str, str]:
    """
    주어진 디렉토리에서 모든 .txt 프롬프트 파일을 불러옵니다.

    Args:
        prompt_dir (Path): 프롬프트 텍스트 파일이 저장된 디렉토리 경로

    Returns:
        Dict[str, str]: 프롬프트 파일 이름을 키, 프롬프트 내용을 값으로 하는 딕셔너리
    """
    prompts = {}
    for file in prompt_dir.glob("*.txt"):
        prompts[file.stem] = file.read_text(encoding="utf-8")
    return prompts


def encode_image(image_path: Path) -> str:
    """
    이미지 파일을 base64 문자열로 인코딩합니다.

    Args:
        image_path (Path): 인코딩할 이미지 파일 경로

    Returns:
        str: base64로 인코딩된 이미지 문자열
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def enrich_prompt_with_data_and_info(prompt: str, data_dict: dict, info_dict: dict) -> str:
    """
    data.json과 info.json을 이용해 프롬프트를 실제 차트 데이터로 보강합니다.

    Args:
        prompt (str): 원본 프롬프트 문자열
        data_dict (dict): .data.json에서 불러온 데이터 딕셔너리
        info_dict (dict): .info.json에서 불러온 정보 딕셔너리

    Returns:
        str: 실제 차트 정보가 추가된 보강된 프롬프트 문자열
    """
    lines = []

    if "chart_type" in info_dict:
        lines.append(f"- 차트 유형: {info_dict['chart_type']}")

    x_col = y_col = y_unit = ""
    for col in data_dict.get("columns", []):
        if col.get("data_type") == "categorical" and not x_col:
            x_col = col["name"]
        elif col.get("data_type") == "numerical" and not y_col:
            y_col = col["name"]
            y_unit = col.get("unit", "")

    x_values = [row.get(x_col, "") for row in data_dict.get("data", [])]
    y_values = [row.get(y_col, "") for row in data_dict.get("data", [])]

    lines.append(f"- X축 ({x_col}): {x_values}")
    lines.append(f"- Y축 ({y_col}): {y_values}" + (f" (단위: {y_unit})" if y_unit else ""))

    enriched = prompt.strip() + "\n\n📊 실제 데이터:\n" + "\n".join(lines)
    return enriched


def evaluate(image_b64: str, prompt: str, api_key: str) -> str:
    """
    base64 인코딩된 이미지와 프롬프트를 OpenAI API에 전송하고 응답을 받아옵니다.

    Args:
        image_b64 (str): base64로 인코딩된 이미지 문자열
        prompt (str): LLM에 보낼 텍스트 프롬프트
        api_key (str): OpenAI API 키

    Returns:
        str: GPT의 응답 텍스트
    """
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 설정
    api_key = "sk-..."
    charts_dir = Path("/home/sdj/ChartGalaxy_test/charts")
    prompt_dir = Path("/home/sdj/ChartGalaxy_test/prompts")
    output_dir = Path("/home/sdj/ChartGalaxy_test/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프롬프트 불러오기
    prompts = load_prompts(prompt_dir)
    chart_files = list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.jpg"))

    # 이미지별로 실행
    for chart_path in chart_files:
        chart_id = chart_path.stem
        output_path = output_dir / f"{chart_id}.json"

        if output_path.exists():
            print(f"이미 완료됨: {output_path.name}")
            continue

        image_b64 = encode_image(chart_path)
        results = {}

        for prompt_name, prompt_text in prompts.items():
            try:
                data_path = chart_path.with_suffix(".data.json")
                info_path = chart_path.with_suffix(".info.json")
                enriched_prompt = prompt_text

                if data_path.exists() and info_path.exists():
                    with open(data_path, "r", encoding="utf-8") as f:
                        data_dict = json.load(f)
                    with open(info_path, "r", encoding="utf-8") as f:
                        info_dict = json.load(f)
                    enriched_prompt = enrich_prompt_with_data_and_info(prompt_text, data_dict, info_dict)

                print(f"실행 중: {chart_id} × {prompt_name}")
                result = evaluate(image_b64, enriched_prompt, api_key)
                results[prompt_name] = result.strip()
                time.sleep(5)

            except Exception as e:
                print(f"오류: {chart_id} × {prompt_name} – {e}")

        if results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "chart_id": chart_id,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            print(f"저장 완료: {output_path.name}")
