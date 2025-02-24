# Embeddings Generator

A Python package for generating text embeddings using transformer models.

## Installation

```bash
pip install git+https://http://github.com/MediaMonitoringAndAnalysis/embeddings_generator.git
```

## Usage

```python
from embeddings_generator import EmbeddingsGenerator

# Initialize with default model
generator = EmbeddingsGenerator()

# Or specify a model
generator = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for a single text
text = "This is a sample text."
embedding = generator(text)
print(f"Embedding shape: {embedding.shape}")

# Generate embeddings for multiple texts
texts = ["This is a sample text.", "Another example sentence."]
embeddings = generator(texts)
print(f"Embeddings shape: {embeddings.shape}")
```

## Project structure

```
├── embeddings_generator/
│   ├── __init__.py                  # Package initialization, exports EmbeddingsGenerator
│   └── generate_embeddings.py       # Main module with EmbeddingsGenerator class
├── pyproject.toml                   # Modern Python packaging configuration
├── setup.py                         # Backward compatibility for pip install
├── README.md                        # Documentation and usage examples
└── LICENSE                          # AGPL-3.0 license file
```

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.
