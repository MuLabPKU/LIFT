from setuptools import setup, find_packages

setup(
    name="lift",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.7.1",
        "transformers==4.53.2",
        "accelerate==1.9.0",
        "peft==0.16.0",
        "bitsandbytes==0.46.1",
        "nltk",
        "vllm==0.10.0",
        "spacy==3.8.7",
        "openai",
        "matplotlib",
        "datasets",
        "liger_kernel",
    ]
)
