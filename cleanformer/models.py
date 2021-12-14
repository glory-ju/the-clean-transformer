from typing import Tuple, Dict
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F

class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        # TODO: implement transformer
        # 학습을 해야하는 레이어? : 임베딩 테이블, 인코더, 디코더 이 3가지를 학습해야 함.
        # (|V|, H)
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        src_ids: (N, L)
        tgt_ids: (N, L)
        return hidden: (N, L, H)
        """
        # --- 임베딩 벡터 불러오기 --- #
        src = self.token_embeddings(src_ids) # (N, L) -> (N, L, H)
        tgt = self.token_embeddings(tgt_ids) # (N, L) -> (N, L, H)
        # --- positional encoding --- #
        # TODO: 나중에
        memory = self.encoder.forward(src) # (N, L, H) -> (N, L, H)
        hidden = self.decoder.forward(tgt, memory) # (N, L, H) -> (N, L, H)
        return hidden


    # 입력 & 레이블을 인자로 받는 함수를 정의해야 한다.
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> dict:
        # batch 속에 무엇이 들어있을까?
        # A : 사용자 마음임. 내가 정의해야 함
        X, Y = batch # (N, 2, 2, L), (N, L)
        # X = 입력
        # encoder 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # decoder 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        hidden = self.forward(src_ids, tgt_ids,
                              src_key_padding_mask, tgt_key_padding_mask) # (N, L, H)
        cls = self.token_embeddings.weight # (|V|, H)
        # 행렬 곱을 해야한다.
        logits = torch.einsum("nlh, vh-> nvl", hidden, cls) # (N, L, H) * (V, H) -> (N, V, L)
        loss = F.cross_entropy(logits, Y) # (N, V, d1=L), (N, d1=L) -> (N,)
        loss = loss.sum() # (N,) -> (,)

        return {
            "loss": loss
        }



class Encoder(torch.nn.Module):
    pass


class Decoder(torch.nn.Module):
    pass