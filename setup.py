from setuptools import setup, find_packages

setup(
    name="embeddings_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tqdm",
        "transformers",
        "numpy"
    ],
    python_requires=">=3.9",
)