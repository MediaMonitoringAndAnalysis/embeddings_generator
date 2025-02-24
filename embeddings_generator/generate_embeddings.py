import torch.nn.functional as F
import torch
from torch import Tensor, float16
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm
import os
import gc


class EmbeddingsGenerator:
    """
    A class for generating embeddings from text using transformer models.
    
    Attributes:
        model_name (str): The name or path of the transformer model to use.
        device (str): The device to run the model on ('cuda' or 'cpu').
        embedding_size (int): The size of the generated embeddings.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the EmbeddingsGenerator with a specified model.
        
        Args:
            model_name (str, optional): The name or path of the transformer model to use.
                If None, uses the MODEL_NAME environment variable or falls back to a default model.
        """
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.embedding_size = self.model.config.hidden_size

    def _average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        Perform average pooling on the model's output.
        
        Args:
            last_hidden_states: The last hidden states from the model.
            attention_mask: The attention mask for the input.
            
        Returns:
            Normalized embeddings in float16 format.
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[
            ..., None
        ].clamp(min=1e-9)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_embeddings_float16 = normalized_embeddings.to(dtype=float16)
        return normalized_embeddings_float16

    @torch.inference_mode()
    def _generate_embeddings_batch(self, batch_texts: List[str]) -> Tensor:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            batch_texts: A list of text strings to generate embeddings for.
            
        Returns:
            Tensor of embeddings for the batch.
        """
        batch_dict = self.tokenizer(
            batch_texts,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        gc.collect()
        return embeddings

    def _generate_embeddings(self, input_texts: List[str], batch_size: int = 64) -> Tensor:
        """
        Generate embeddings for a list of input texts.
        
        Args:
            input_texts: List of input text sentences.
            batch_size: Size of the batches for processing.
            
        Returns:
            Tensor containing embeddings for the input texts.
        """
        n_input_texts = len(input_texts)
        input_sentences = np.array(input_texts)
        sentences_lengths = np.array(
            [len(sentence.split()) for sentence in input_texts]
        )
        sorted_indices = np.argsort(sentences_lengths)
        sorted_sentences = input_sentences[sorted_indices].tolist()

        sorted_inputs_embeddings = torch.zeros(
            (n_input_texts, self.embedding_size), dtype=float16
        )
        with torch.inference_mode():
            if n_input_texts < batch_size:
                sorted_inputs_embeddings = self._generate_embeddings_batch(sorted_sentences).cpu()
            else:
                for i in tqdm(range(0, n_input_texts, batch_size), desc="Generating embeddings"):
                    sorted_batch = sorted_sentences[i : i + batch_size]
                    embeddings_batch = self._generate_embeddings_batch(sorted_batch).cpu()
                    sorted_inputs_embeddings[i : i + batch_size] = embeddings_batch

        final_embeddings = torch.zeros_like(sorted_inputs_embeddings, dtype=float16)
        final_embeddings[sorted_indices] = sorted_inputs_embeddings

        return final_embeddings

    def generate(self, texts: Union[str, List[str]], batch_size: int = 64) -> Tensor:
        """
        Generate embeddings for a single text or a list of texts.
        
        Args:
            texts: A single text string or a list of text strings.
            batch_size: Size of the batches for processing.
            
        Returns:
            Tensor containing embeddings for the input text(s).
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self._generate_embeddings(texts, batch_size)
        
        return embeddings
    
    def __call__(self, input_texts: Union[str, List[str]], batch_size: int = 64) -> Tensor:
        """
        Generate embeddings for a single text or a list of texts.
        
        Args:
            input_texts: A single text string or a list of text strings.
            batch_size: Size of the batches for processing.
            
        Returns:
            Tensor containing embeddings for the input text(s).
        """
        return self.generate(input_texts, batch_size)


# Example usage
# if __name__ == "__main__":
#     # Example usage
#     generator = EmbeddingsGenerator()
#     texts = ["This is a sample text.", "Another example sentence."]
#     embeddings = generator(texts)
#     print(f"Generated embeddings shape: {embeddings.shape}")
