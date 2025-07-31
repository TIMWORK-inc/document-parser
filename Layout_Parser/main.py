import importlib
import argparse
from config.config import Config

# 어댑터 이름 → 모듈 경로 매핑
ADAPTERS = {
    'docbank': 'parsers.docbank_adapter',
    'layoutparser': 'parsers.layoutparser_adapter',
    'pdfplumber': 'parsers.pdfplumber_adapter',
    'publaynet': 'parsers.publaynet_adapter',
}

def main():
    parser = argparse.ArgumentParser(description="문서 레이아웃 분석 어댑터 실행기")
    parser.add_argument('--adapter', required=True, choices=ADAPTERS.keys(),
                        help='사용할 어댑터 이름')
    parser.add_argument('--file', help='사용자 정의 문서 경로 (선택 사항)')

    args = parser.parse_args()

    # 사용자 입력이 있을 경우 Config 값 덮어쓰기
    if args.file:
        Config.DOCUMENT_PATH = args.file

    # 어댑터 모듈 import 및 실행
    module = importlib.import_module(ADAPTERS[args.adapter])
    adapter = module.Adapter()
    adapter.parse(Config.DOCUMENT_PATH)

if __name__ == '__main__':
    main()
