import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    PROJECT_NAME = "Receipt Analyzer API"
    PROJECT_VERSION = "1.0.0"
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this")
    API_USERNAME = os.getenv("API_USERNAME", "admin")
    API_PASSWORD = os.getenv("API_PASSWORD", "38digital123")
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # File Upload
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB
    
settings = Settings()

CONFIG = {
    'SUPPORTED_LANGUAGES': ['eng', 'chi_sim', 'chi_tra'],
    'SUPPORTED_CURRENCIES': ['¥', 'RMB', '元', '$', 'HKD', 'MOP', 'CNY'],
    'OUTPUT_DIR': 'output',
    'LOGS_DIR': 'logs',
    'MIN_CONFIDENCE': 0.5,
}

KEYWORDS = {
    'total': ['总额', '合计', 'total', '总计', '金额', '应付', '实付', '消费金额'],
    'date': ['日期', '时间', 'date', '开票日期', '消费日期'],
    'merchant': ['商户', '商家', '店名', '商户名称', '公司名称'],
    'currency': CONFIG['SUPPORTED_CURRENCIES'],
    'tax': ['税额', 'TAX', '税金', '增值税'],
}

# Add Chinese date formats
DATE_FORMATS = {
    'chinese': [
        '%Y年%m月%d日',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ],
    'western': [
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y-%m-%d',
    ]
}