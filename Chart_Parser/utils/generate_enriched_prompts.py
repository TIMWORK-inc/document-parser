import json
from pathlib import Path

def enrich_prompt_with_data_and_info(prompt: str, data_dict: dict, info_dict: dict) -> str:
    """
    주어진 프롬프트에 차트의 데이터와 정보를 추가하여 보강된 프롬프트를 생성합니다.

    Args:
        prompt (str): 기본 프롬프트 문자열.
        data_dict (dict): 차트의 .data.json 파일에서 불러온 데이터 딕셔너리.
        info_dict (dict): 차트의 .info.json 파일에서 불러온 메타정보 딕셔너리.

    Returns:
        str: 데이터와 정보가 포함된 보강된 프롬프트 문자열.
    """
    lines = []

    # 차트 유형 정보 추가
    if "chart_type" in info_dict:
        lines.append(f"- 차트 유형: {info_dict['chart_type']}")

    x_col = y_col = y_unit = ""
    for col in data_dict.get("columns", []):
        if col.get("data_type") == "categorical" and not x_col:
            x_col = col["name"]
        elif col.get("data_type") == "numerical" and not y_col:
            y_col = col["name"]
            y_unit = col.get("unit", "")

    # 실제 데이터 값 추가
    if x_col and y_col:
        try:
            x_values = [row[x_col] for row in data_dict["data"]]
            y_values = [row[y_col] for row in data_dict["data"]]
            lines.append(f"- X축 ({x_col}): {x_values}")
            lines.append(f"- Y축 ({y_col}): {y_values}" + (f" (단위: {y_unit})" if y_unit else ""))
        except KeyError:
            lines.append("일부 데이터 행에 누락된 값이 있습니다.")
    else:
        lines.append("유효한 X축 또는 Y축 컬럼을 찾을 수 없습니다.")

    enriched = prompt.strip() + "\n\n실제 데이터:\n" + "\n".join(lines)
    return enriched


def generate_enriched_prompts(prompt_dir: Path, chart_dir: Path, output_dir: Path):
    """
    프롬프트 템플릿과 차트 메타데이터를 기반으로 보강된 프롬프트를 생성하여 저장합니다.

    Args:
        prompt_dir (Path): 프롬프트 템플릿(.txt)들이 저장된 디렉토리 경로.
        chart_dir (Path): .data.json과 .info.json이 포함된 차트 디렉토리 경로.
        output_dir (Path): 보강된 프롬프트를 저장할 출력 디렉토리 경로.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프롬프트 템플릿 불러오기
    prompt_dict = {}
    for file in prompt_dir.glob("*.txt"):
        prompt_dict[file.stem] = file.read_text(encoding="utf-8")

    # 각 차트에 대해 보강된 프롬프트 생성
    for info_file in chart_dir.glob("*.info.json"):
        chart_id = info_file.stem.replace(".info", "")
        data_file = chart_dir / f"{chart_id}.data.json"
        if not data_file.exists():
            print(f"{chart_id} - .data.json 없음")
            continue

        with open(info_file, encoding="utf-8") as f:
            info = json.load(f)
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        for name, template in prompt_dict.items():
            enriched = enrich_prompt_with_data_and_info(template, data, info)
            output_path = output_dir / f"{chart_id}_{name}.txt"
            output_path.write_text(enriched, encoding="utf-8")
            print(f"생성됨: {output_path.name}")


# 실행
generate_enriched_prompts(
    Path("/home/sdj/ChartGalaxy_test/prompts"),
    Path("/home/sdj/ChartGalaxy_test/charts"),
    Path("/home/sdj/ChartGalaxy_test/enriched_prompts")
)
