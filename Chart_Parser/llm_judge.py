import json
from pathlib import Path
from openai import OpenAI
from typing import Dict

def load_chart_data(chart_id: str, charts_dir: Path):
    with open(charts_dir / f"{chart_id}.data.json", "r", encoding="utf-8") as f:
        chart_data = json.load(f)
    with open(charts_dir / f"{chart_id}.info.json", "r", encoding="utf-8") as f:
        chart_info = json.load(f)
    return chart_data, chart_info

def load_llm_a_outputs(chart_id: str, outputs_dir: Path) -> Dict[str, str]:
    path = outputs_dir / f"{chart_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError(f"'results' 키가 없습니다: {chart_id}")
    return data["results"]

def build_llm_b_prompt(chart_data, chart_info, prompt_type: str, llm_a_output: str) -> str:
    return f"""
당신은 차트 생성기에서 사용된 데이터와 정보를 보고, 차트를 분석한 결과가 올바른지 판단하는 전문가입니다.

[Prompt Type: {prompt_type}]

다음은 차트를 구성한 데이터 및 메타정보입니다:
[CHART DATA]
{json.dumps(chart_data, ensure_ascii=False)}

[CHART INFO]
{json.dumps(chart_info, ensure_ascii=False)}

다음은 이 차트에 대해 LLM A가 생성한 응답입니다:
[LLM A OUTPUT]
"{llm_a_output}"

질문: LLM A의 응답은 주어진 chart_data 및 chart_info를 기반으로 할 때 정확합니까?
- 정확하면 "Yes"라고 답하고,
- 부정확하면 "No"라고 답하며 그 이유를 간단히 써주세요.
"""

def run_llm_b(prompt: str) -> str:
    client = OpenAI(    api_key = "sk-...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 차트 평가 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def main():
    charts_dir = Path("/home/sdj/ChartGalaxy_test/charts")
    outputs_dir = Path("/home/sdj/ChartGalaxy_test/outputs")
    save_dir = Path("/home/sdj/ChartGalaxy_test/judged_outputs")
    save_dir.mkdir(exist_ok=True)

    for file in outputs_dir.glob("*.json"):
        chart_id = file.stem
        try:
            chart_data, chart_info = load_chart_data(chart_id, charts_dir)
            llm_a_outputs = load_llm_a_outputs(chart_id, outputs_dir)

            judged_result = {}
            for prompt_type, llm_a_output in llm_a_outputs.items():
                prompt = build_llm_b_prompt(chart_data, chart_info, prompt_type, llm_a_output)
                judgment = run_llm_b(prompt)
                judged_result[prompt_type] = {
                    "llm_a_output": llm_a_output,
                    "llm_b_judgment": judgment
                }

            # Save the judged result per chart
            with open(save_dir / f"{chart_id}.judged.json", "w", encoding="utf-8") as f:
                json.dump({
                    "chart_id": chart_id,
                    "judged_results": judged_result
                }, f, ensure_ascii=False, indent=2)

            print(f"{chart_id} 평가 완료")
        except Exception as e:
            print(f"{chart_id} 실패: {e}")

if __name__ == "__main__":
    main()
