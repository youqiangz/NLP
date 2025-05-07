from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="ner-service",
    version="1.0.0",
    description='NER Service',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "torch>=1.9.0",
        "transformers>=4.12.3"
    ],
    entry_points={
        "console_scripts": [
            "ner-service=ner_service.cli:main"
        ]
    }
)