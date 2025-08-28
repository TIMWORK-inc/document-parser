# logger/logger_config.py

import logging
import os

class DocumentParserLogger:
    _initialized = False

    @classmethod
    def initialize(cls, log_dir="logs", log_level=logging.DEBUG):
        if cls._initialized:
            return

        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        file_path = os.path.join(log_dir, "document_parser.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        logging.basicConfig(level=log_level, handlers=[console_handler, file_handler])

        cls._initialized = True

    @classmethod
    def get_logger(cls, module_name):
        if not cls._initialized:
            cls.initialize()
        return logging.getLogger(module_name)
