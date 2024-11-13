
from setuptools import setup, find_packages

setup(
    name="receipt-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "python-doctr>=0.6.0",
        "matplotlib>=3.4.3",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=0.19.0",
        "fuzzywuzzy>=0.18.0",
    ],
    author="38 Digits",
    author_email="sameer@dataahunt.app",
    description="A receipt analysis system using OCR and AI",
    keywords="ocr,receipt,analysis,ai",
    python_requires=">=3.8",
)