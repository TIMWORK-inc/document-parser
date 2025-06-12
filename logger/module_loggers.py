# logger/module_loggers.py

from .logger_config import DocumentParserLogger

# 모듈별 로거 정의
LayoutDetectionLogger = DocumentParserLogger.get_logger("LayoutDetection")
OCRLogger = DocumentParserLogger.get_logger("OCR")
TableParserLogger = DocumentParserLogger.get_logger("TableParser")
IntegrationLogger = DocumentParserLogger.get_logger("Integration")
