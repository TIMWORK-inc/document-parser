import json
import argparse
from pathlib import Path
from typing import Dict
import openai

def load_chart_data(chart_id: str, charts_dir: Path):
    with open(charts_dir / f"{chart_id}.data.json", "r", encoding="utf-8") as f:
        chart_data = json.load(f)
    with open(charts_dir / f"{chart_id}.info.json", "r", encoding="utf-8") as f:
        chart_info = json.load(f)
    return chart_data, chart_info

def load_llm_a_outputs(chart_id: str, outputs_dir: Path) -> Dict[str, str]:
    path = outputs_dir / f"{chart_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"❌ LLM A 결과 파일이 존재하지 않습니다: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    # 리스트로 되어 있으면 딕셔너리로 변환
    if isinstance(results, list):
        print("📌 'results'는 리스트 형식입니다. 딕셔너리로 변환 중...")
        results = {
            item["prompt_type"]: item["result"]
            for item in results
            if "prompt_type" in item and "result" in item
        }

    if not results:
        print("⚠️ 'results'가 비어 있습니다. LLM A 결과가 없거나 구조가 잘못됐을 수 있습니다.")
    else:
        print(f"📄 LLM A 결과 로딩 완료. 프롬프트 유형: {list(results.keys())}")

    return results

def build_llm_b_prompt(chart_data, chart_info, prompt_type: str, llm_a_output: str) -> str:
    return f"""
당신은 차트를 생성하는 데 사용된 데이터와 차트 메타 정보를 기반으로,
다음 해석이 정확한지 검증하는 역할을 맡았습니다.

[차트 생성 데이터]
{json.dumps(chart_data, ensure_ascii=False, indent=2)}

[차트 메타 정보]
{json.dumps(chart_info, ensure_ascii=False, indent=2)}

[LLM A의 차트 해석 결과]
{llm_a_output}

[검증 기준: {prompt_type}]
위 내용을 바탕으로 해석 결과가 정확한지 평가해주세요.
가능하다면 '정확함', '부분적으로 정확함', '부정확함' 중 하나로 판단하고 간단한 근거를 제시해주세요.
"""

def evaluate_with_llm_b(prompt: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM B 평가 시스템")
    parser.add_argument("--chart_id", type=str, required=True, help="차트 ID (예: PMC123456)")
    parser.add_argument("--charts_dir", type=Path, required=True, help=".data.json, .info.json이 저장된 디렉토리")
    parser.add_argument("--outputs_dir", type=Path, required=True, help="LLM A 결과가 저장된 디렉토리")
    parser.add_argument("--prompt_type", type=str, default=None, help="프롬프트 유형 필터 (예: chart_analysis)")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API 키")
    args = parser.parse_args()

    print(f"🔍 차트 ID: {args.chart_id}")
    print(f"📁 charts_dir: {args.charts_dir}")
    print(f"📁 outputs_dir: {args.outputs_dir}")
    print(f"🔎 평가 대상 prompt_type: {args.prompt_type or '모두'}")

    # 데이터 로드
    chart_data, chart_info = load_chart_data(args.chart_id, args.charts_dir)
    llm_a_outputs = load_llm_a_outputs(args.chart_id, args.outputs_dir)

    if not llm_a_outputs:
        print("⚠️ 평가할 LLM A 결과가 없습니다. 스크립트를 종료합니다.")
        exit(1)

    # 평가 시작
    for prompt_type, llm_a_output in llm_a_outputs.items():
        if args.prompt_type and prompt_type != args.prompt_type:
            continue
        print(f"\n🧪 [{prompt_type}] 항목 평가 중...")
        prompt = build_llm_b_prompt(chart_data, chart_info, prompt_type, llm_a_output)
        result = evaluate_with_llm_b(prompt, args.api_key)
        print(f"✅ 평가 결과:\n{result}")
