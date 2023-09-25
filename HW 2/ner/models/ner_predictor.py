from dataclasses import dataclass

import torch

from ner.nn.embeddings.embedding import TokenEmbedding
from ner.nn.models import ffnn, rnn
from ner.nn.module import Module


class NERPredictor(Module):
    def __init__(self, model: str = "ffnn", **kwargs):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=kwargs.get("vocab_size"),
            embedding_dim=kwargs.get("embedding_dim"),
            padding_idx=kwargs.get("padding_idx", 0),
        )

        if model == "ffnn":
            self.model = ffnn.FFNN(
                embedding_dim=kwargs.get("embedding_dim"),
                hidden_dim=kwargs.get("hidden_dim"),
                output_dim=kwargs.get("output_dim"),
                num_layers=kwargs.get("num_layers", 1),
            )
        elif model == "rnn":
            self.model = rnn.RNN(
                embedding_dim=kwargs.get("embedding_dim"),
                hidden_dim=kwargs.get("hidden_dim"),
                output_dim=kwargs.get("output_dim"),
                num_layers=kwargs.get("num_layers", 1),
                bias=kwargs.get("bias", True),
                nonlinearity=kwargs.get("nonlinearity", "tanh"),
            )
        else:
            raise ValueError(f"model not supported")

    def forward(self, input_ids: torch.Tensor):
        # (batch_size, max_length) -> (batch_size, max_length, embedding_dim)
        embeddings = self.embedding(input_ids)
        # (batch_size, max_length, embedding_dim) -> (batch_size, max_length, output_dim)
        outputs = self.model(embeddings)
        return outputs


if __name__ == "__main__":

    @dataclass
    class TestConfig:
        batch_size = 2
        max_length = 6

        vocab_size = 10
        embedding_dim = 5
        padding_idx = 9

        hidden_dim = 3
        output_dim = 3
        num_layers = 2
        bias = True
        nonlinearity = "tanh"

    test_config = TestConfig()
    test_ner_predictor = NERPredictor(
        vocab_size=test_config.vocab_size,
        embedding_dim=test_config.embedding_dim,
        padding_idx=test_config.padding_idx,
        hidden_dim=test_config.hidden_dim,
        output_dim=test_config.output_dim,
        num_layers=test_config.num_layers,
        bias=test_config.bias,
        nonlinearity=test_config.nonlinearity,
    )
    test_ner_predictor.print_params()

    test_outputs = test_ner_predictor(
        input_ids=torch.randint(0, test_config.vocab_size, (test_config.batch_size, test_config.max_length))
    )
    assert test_outputs.shape == (test_config.batch_size, test_config.max_length, test_config.output_dim)
