from typing import OrderedDict, Optional, Tuple

from dataclasses import dataclass
import json
import math

import torch
from transformers import AutoModel, BertTokenizer, BertConfig
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


@dataclass
class BertConfig:
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: Optional[float] = None
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 128
    initializer_range: float = 0.02
    intermediate_size: int = 512
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    model_type: str = "bert"
    num_attention_heads: int = 2
    num_hidden_layers: int = 2
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    transformers_version: str = "4.36.2"
    type_vocab_size: int = 2
    use_cache: bool = True
    vocab_size: int = 30522

    @staticmethod
    def from_pretrained_model_or_path(pretrained_model_name_or_path: str) -> BertConfig:
        resolved_archive_file = cached_file(
            path_or_repo_id=pretrained_model_name_or_path,
            filename=CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        config_content = json.load(open(resolved_archive_file))
        return BertConfig(**config_content)


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertEmbeddings(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)

        word_embedding = self.word_embeddings(input_ids)
        token_type_embedding = self.token_type_embeddings(token_type_ids)
        position_embedding = self.position_embeddings(position_ids)

        # Combine all embeddings
        embedding = word_embedding + position_embedding + token_type_embedding

        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding
    

class BertSelfAttention(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.head_size * self.num_attention_heads

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.transpose_for_scores(self.query(inputs))
        k = self.transpose_for_scores(self.key(inputs))
        v = self.transpose_for_scores(self.value(inputs))

        # Attetnion score
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)

        # If `attention_mask` is None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :].to(dtype=inputs.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(inputs.dtype).min
            attention_scores = attention_scores + extended_attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_outputs = torch.matmul(attention_probs, v)

        # Merge the head weights, (batch_size, seq_len, head_size)
        attention_outputs = attention_outputs.permute(0, 2, 1, 3).contiguous()
        attention_outputs = attention_outputs.view(*attention_outputs.shape[:2], -1)
        return attention_outputs
    

class BertSelfOutput(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
        )
        self.LayerNorm = torch.nn.LayerNorm(normalized_shape=(config.hidden_size,), eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(p=config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(hidden_states)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(outputs + inputs)
        return outputs


class BertAttention(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = BertSelfAttention(config=config)
        self.output = BertSelfOutput(config=config)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.self(inputs, attention_mask=attention_mask)
        outputs = self.output(hidden_states=hidden_states, inputs=inputs)
        return outputs
    

class BertIntermediate(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
        )
        self.intermediate_act_fn = torch.nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        outputs = self.intermediate_act_fn(outputs)
        return outputs
    
class BertOutput(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
        )
        self.LayerNorm = torch.nn.LayerNorm(normalized_shape=(config.hidden_size), eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(p=config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        outputs = self.LayerNorm(hidden_states + inputs)
        return outputs


class BertPooler(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
        )
        self.activation = torch.nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cls_token_tensor = inputs[:, 0]
        pooler_outputs = self.dense(cls_token_tensor)
        pooler_outputs = self.activation(pooler_outputs)
        return pooler_outputs


class BertLayer(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config=config)
        self.intermediate = BertIntermediate(config=config)
        self.output = BertOutput(config=config)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> None:
        attention_outputs = self.attention(inputs=inputs, attention_mask=attention_mask)
        intermediate_outputs = self.intermediate(inputs=attention_outputs)
        outputs = self.output(hidden_states=intermediate_outputs, inputs=attention_outputs)
        return outputs


class BertEncoder(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [BertLayer(config=config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutputWithPastAndCrossAttentions:
        for layer_module in self.layer:
            inputs = layer_module(inputs=inputs, attention_mask=attention_mask)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=inputs)
    

class BertModel(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = BertEncoder(config=config)
        self.pooler = BertPooler(config=config)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutputWithPoolingAndCrossAttentions:
        embedding = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoded = self.encoder(embedding, attention_mask=attention_mask)
        pooler_output = self.pooler(encoded.last_hidden_state)
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=encoded.last_hidden_state,
            pooler_output=pooler_output,
        )
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> "BertModel":
        """Load pretrained weights from HuggingFace into model.
        
        Args:
            pretrained_model_name_or_path: One of
                * "prajjwal1/bert-tiny"
                ...

        Returns:
            model: BertModel model with weights loaded
        """

        def load_state_dict_hf(path_or_repo_id: str) -> OrderedDict:
            resolved_archive_file = cached_file(
                path_or_repo_id=path_or_repo_id,
                filename=WEIGHTS_NAME,
            )
            return torch.load(resolved_archive_file, weights_only=True)

        # Load config
        config = BertConfig.from_pretrained_model_or_path(pretrained_model_name_or_path=pretrained_model_name_or_path)

        # Load weights
        new_state_dict = {}
        state_dict = load_state_dict_hf(pretrained_model_name_or_path)

        for key in state_dict:
            if "cls" in key or "position_ids" in key:
                continue

            new_key = key.replace("bert.", "")
            new_state_dict[new_key] = state_dict[key]

        # Load model
        model = BertModel(config=config)
        model.load_state_dict(new_state_dict)

        return model


if __name__ == "__main__":
    # Init
    pretrained_model_name_or_path = "prajjwal1/bert-tiny"

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

    # Model
    my_model = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()
    hf_model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()

    # Data
    sentences = [
        ("Today is a nice day", "I want to go to play"),
        ("Hello", "Nice to meet you too")
    ]

    inputs = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True, return_tensors="pt")

    print(my_model(**inputs).last_hidden_state[0][0][0])
    print(hf_model(**inputs).last_hidden_state[0][0][0])
    print(my_model(**inputs).last_hidden_state[0][0][0] == hf_model(**inputs).last_hidden_state[0][0][0])
    print(torch.allclose(my_model(**inputs).last_hidden_state, hf_model(**inputs).last_hidden_state, atol=1e-4))
