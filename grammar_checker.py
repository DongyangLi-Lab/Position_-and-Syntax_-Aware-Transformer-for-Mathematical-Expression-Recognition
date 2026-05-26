import torch
import torch.nn as nn


class GrammarChecker(nn.Module):
    """
    GrammarChecker(prefix_ids, candidate_ids) -> logits

    输入:
        prefix_ids: LongTensor [B, T]
            当前已经生成的前缀 token ids，例如 [_START_, "\\frac", "{"]
        candidate_ids: LongTensor [B]
            候选下一个 token id，例如 "x" 或 "}"

    输出:
        logits: FloatTensor [B]
            未经过 sigmoid 的合法性分数。
            使用 torch.sigmoid(logits) 后得到 0~1 合法置信度。
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_size + emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, prefix_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        prefix_ids: [B, T]
        candidate_ids: [B]
        return logits: [B]
        """
        if prefix_ids.dim() != 2:
            raise ValueError(f"prefix_ids should be [B, T], got shape {tuple(prefix_ids.shape)}")
        if candidate_ids.dim() != 1:
            raise ValueError(f"candidate_ids should be [B], got shape {tuple(candidate_ids.shape)}")
        if prefix_ids.size(0) != candidate_ids.size(0):
            raise ValueError("Batch size mismatch between prefix_ids and candidate_ids")

        mask = prefix_ids.ne(self.pad_id)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()

        emb = self.embedding(prefix_ids)  # [B, T, E]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)

        if self.bidirectional:
            # h_n: [num_layers * 2, B, H]
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            prefix_state = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            prefix_state = h_n[-1]  # [B, H]

        cand_emb = self.embedding(candidate_ids)  # [B, E]
        joint = torch.cat([prefix_state, cand_emb], dim=-1)

        logits = self.classifier(joint).squeeze(-1)
        return logits

    @torch.no_grad()
    def score(self, prefix_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        返回 0~1 合法置信度。
        """
        self.eval()
        return torch.sigmoid(self.forward(prefix_ids, candidate_ids))
