import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from grammar_checker import GrammarChecker

from Calcu import TokenEmbedding, PositionalEncoding, create_mask


class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    """
    Drop-in replacement for nn.TransformerDecoderLayer that can STORE cross-attn weights
    without changing module names / state_dict keys.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # runtime flags / buffers
        self.capture_attn = False
        self.last_cross_attn = None  # will store [B, nhead, T, S] if capture_attn=True

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False
    ):
        # This forward is based on PyTorch's TransformerDecoderLayer forward,
        # except we set need_weights=True for cross-attn when capture_attn is enabled,
        # and store the weights in self.last_cross_attn.
        x = tgt

        # self-attn block
        x2 = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # cross-attn block (multihead_attn)
        if self.capture_attn:
            x2, attn = self.multihead_attn(
                x, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=True,
                average_attn_weights=False  # -> [B, nhead, T, S]
            )
            self.last_cross_attn = attn
        else:
            x2 = self.multihead_attn(
                x, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False
            )[0]
            self.last_cross_attn = None

        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # FFN block
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)

        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        num_decoder_layers: int,
        emb_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        n_head=8,
        dropout: float = 0.1,
        softmax=False,
        vocab=None,
        max_len=150,
        beam_size: int = 1,
        length_penalty: float = 0.2,
        grammar_checker_path: str = "./checkpoints/grammar_checker/grammar_checker_best.pt",
        grammar_alpha: float = 0.015,
        grammar_log_min: float = -2.0,
        grammar_log_max: float = 0.0,
        grammar_eps: float = 1e-6
    ):
        super().__init__()
        self._printed_shape = False

        decoder_layer = TransformerDecoderLayerWithAttn(
            d_model=emb_size,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        # IMPORTANT: keep this attribute name to match old checkpoints
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        if softmax:
            self.classifier = nn.Sequential(nn.Linear(emb_size, tgt_vocab_size), nn.LogSoftmax(dim=-1))
        else:
            self.classifier = nn.Sequential(nn.Linear(emb_size, tgt_vocab_size))

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.train_generator = False
        self.vocab = vocab
        self.max_len = max_len

        # Inference-only decoding parameters.
        # beam_size=1 is equivalent to greedy decoding.
        self.beam_size = beam_size
        self.length_penalty = length_penalty

        # Grammar-aware beam scoring.
        # IMPORTANT:
        # Do NOT assign GrammarChecker to self.grammar_checker directly.
        # nn.Module assigned to self.xxx is registered as a child module and
        # will pollute DecoderTransformer.state_dict(), breaking old MER checkpoints.
        self.grammar_checker_path = grammar_checker_path
        self.grammar_alpha = float(grammar_alpha)
        self.grammar_log_min = float(grammar_log_min)
        self.grammar_log_max = float(grammar_log_max)
        self.grammar_eps = float(grammar_eps)
        self._grammar_checker_holder = [None]  # plain Python list: not registered by nn.Module

        # Always load the grammar checker when a checkpoint path is provided.
        # This keeps the decoding path identical for grammar_alpha=0, 1e-6, 0.1, etc.
        # When grammar_alpha == 0, the grammar term is still computed but multiplied by 0.
        self._load_grammar_checker(grammar_checker_path)

        # will hold last-layer cross-attn for the latest _forward call
        self.last_cross_attn = None  # [B, nhead, T, S] or None

    def _set_capture_attn(self, enabled: bool):
        for layer in self.transformer_decoder.layers:
            if hasattr(layer, "capture_attn"):
                layer.capture_attn = enabled
                if not enabled:
                    layer.last_cross_attn = None

    def _forward(self, features, targets, target_mask=None, target_padding_mask=None, return_attn: bool = False):
        if not self._printed_shape:
            print("[DEBUG] features.shape:", tuple(features.shape))
            print("[DEBUG] targets.shape :", tuple(targets.shape))
            self._printed_shape = True

        # enable/disable capturing only for this call
        self._set_capture_attn(return_attn)

        if self.train_generator:
            with torch.no_grad():
                tgt_emb = self.positional_encoding(self.tgt_tok_emb(targets))
                outs = self.transformer_decoder(
                    tgt_emb,
                    features,
                    tgt_mask=target_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=target_padding_mask,
                    memory_key_padding_mask=None
                )
        else:
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(targets))
            outs = self.transformer_decoder(
                tgt_emb,
                features,
                tgt_mask=target_mask,
                memory_mask=None,
                tgt_key_padding_mask=target_padding_mask,
                memory_key_padding_mask=None
            )

        logits = self.classifier(outs)

        # collect last layer cross-attn if requested
        if return_attn:
            last_layer = self.transformer_decoder.layers[-1]
            self.last_cross_attn = getattr(last_layer, "last_cross_attn", None)
            return logits, self.last_cross_attn

        self.last_cross_attn = None
        return logits


    def _load_grammar_checker(self, checkpoint_path: str):
        """
        Load separately trained GrammarChecker without registering it as a child module.
        This keeps old MER checkpoints loadable.
        """
        if GrammarChecker is None:
            print("[WARN] GrammarChecker class not found; grammar-aware beam is disabled.")
            self._grammar_checker_holder[0] = None
            return

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[WARN] Grammar checker checkpoint not found: {checkpoint_path}; grammar-aware beam is disabled.")
            self._grammar_checker_holder[0] = None
            return

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            args = ckpt.get("args", {})
            id2token = ckpt.get("id2token", None)
            special_ids = ckpt.get("special_ids", {})

            vocab_size = len(id2token) if id2token is not None else self.classifier[0].out_features
            pad_id = int(special_ids.get("pad_id", self.vocab.token2id.get("_PAD_", 0)))

            checker = GrammarChecker(
                vocab_size=vocab_size,
                emb_size=int(args.get("emb_size", 256)),
                hidden_size=int(args.get("hidden_size", 256)),
                num_layers=int(args.get("num_layers", 1)),
                dropout=float(args.get("dropout", 0.1)),
                pad_id=pad_id,
                bidirectional=bool(args.get("bidirectional", False)),
            )

            state_dict = ckpt.get("model_state_dict", ckpt)
            checker.load_state_dict(state_dict, strict=True)
            checker.eval()
            for p in checker.parameters():
                p.requires_grad_(False)

            self._grammar_checker_holder[0] = checker
            print(
                "[INFO] Loaded GrammarChecker for grammar-aware beam: "
                f"{checkpoint_path}; alpha={self.grammar_alpha}, "
                f"log_clamp=[{self.grammar_log_min}, {self.grammar_log_max}]"
            )
        except Exception as exc:
            print(f"[WARN] Failed to load grammar checker from {checkpoint_path}: {exc}")
            print("[WARN] Grammar-aware beam is disabled; using ordinary beam search.")
            self._grammar_checker_holder[0] = None

    def _grammar_bonus(self, prefix_tokens, candidate_ids, device):
        """
        Finite grammar-aware soft penalty for candidate tokens.
        Returns [K], already multiplied by grammar_alpha.

        The grammar checker is called regardless of grammar_alpha so that all alpha values
        follow the same beam-search path. If grammar_alpha == 0, the final returned
        penalty is numerically zero after multiplication.
        """
        if candidate_ids.numel() == 0:
            return torch.zeros(candidate_ids.size(0), dtype=torch.float, device=device)

        checker = self._grammar_checker_holder[0]
        if checker is None:
            return torch.zeros(candidate_ids.size(0), dtype=torch.float, device=device)

        checker.to(device)
        checker.eval()

        with torch.no_grad():
            prefix_1d = prefix_tokens.squeeze(1).to(device).long()  # [T]
            prefixes = prefix_1d.unsqueeze(0).repeat(candidate_ids.size(0), 1)  # [K, T]
            candidate_ids = candidate_ids.to(device).long()  # [K]

            logits = checker(prefixes, candidate_ids)
            probs = torch.sigmoid(logits).clamp(min=self.grammar_eps, max=1.0)
            grammar_log = torch.log(probs)
            grammar_log = torch.clamp(grammar_log, min=self.grammar_log_min, max=self.grammar_log_max)
            return self.grammar_alpha * grammar_log

    def _sequence_score(self, score: float, generated_len: int) -> float:
        """
        Optional length normalization for comparing finished beams.
        length_penalty=0 keeps the original accumulated log-prob score.
        """
        if self.length_penalty <= 0:
            return score
        return score / (max(generated_len, 1) ** self.length_penalty)

    def _beam_search_decode_one(self, feature_i, device):
        """
        Beam search for a single sample.

        Returns:
            best_tokens: LongTensor [T+1, 1], including _START_ and generated tokens.
                         Generated tokens usually end with _END_ unless max_len is reached.
        """
        start_id = self.vocab.token2id['_START_']
        end_id = self.vocab.token2id['_END_']
        pad_id = self.vocab.token2id['_PAD_']

        beam_size = max(int(self.beam_size), 1)

        # Each beam is (tokens, accumulated_score, finished)
        # tokens includes _START_.
        start_tokens = torch.tensor([[start_id]], dtype=torch.long, device=device)
        beams = [(start_tokens, 0.0, False)]
        completed = []

        # max_len follows the old behavior: stop when tgt_inputs.size(0) >= self.max_len.
        # Since tokens includes _START_, at most max_len-1 new tokens are generated.
        for _ in range(self.max_len - 1):
            candidates = []

            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue

                tgt_masks, tgt_padding_masks = create_mask(tokens, device, pad_id)
                outputs = self._forward(
                    feature_i,
                    tokens,
                    tgt_masks,
                    tgt_padding_masks,
                    return_attn=False
                )

                # outputs[-1, 0] is the distribution for the next token after this prefix.
                log_probs = F.log_softmax(outputs[-1, 0, :], dim=-1)
                top_log_probs, top_ids = torch.topk(log_probs, k=min(beam_size, log_probs.size(-1)))

                grammar_bonuses = self._grammar_bonus(tokens, top_ids, device)

                for log_prob, token_id, grammar_bonus in zip(top_log_probs, top_ids, grammar_bonuses):
                    next_token = token_id.view(1, 1)
                    next_tokens = torch.cat([tokens, next_token], dim=0)
                    next_score = score + float(log_prob.item()) + float(grammar_bonus.item())
                    is_finished = int(token_id.item()) == end_id

                    if is_finished:
                        completed.append((next_tokens, next_score, True))
                    else:
                        candidates.append((next_tokens, next_score, False))

            if not candidates:
                break

            # Keep only top K active candidates after optional length normalization.
            candidates = sorted(
                candidates,
                key=lambda item: self._sequence_score(item[1], item[0].size(0) - 1),
                reverse=True
            )
            beams = candidates[:beam_size]

            # If all retained beams are finished, we can stop.
            if all(item[2] for item in beams):
                break

        final_pool = completed if completed else beams
        best_tokens, _, _ = max(
            final_pool,
            key=lambda item: self._sequence_score(item[1], item[0].size(0) - 1)
        )
        return best_tokens

    def forward(self, features, labels, return_attn: bool = False):
        """
        Training: returns logits (same behavior as original).
        Eval: uses beam search but still returns list of logits, so old model.py can keep:
              torch.argmax(out_i, -1).squeeze(dim=1)

        If return_attn=True (eval only): returns (out_list, attn_info)
          attn_info["cross_attn_last"][i] is [nhead, T, S] for sample i.

        Notes:
          - beam_size=1 is equivalent to greedy decoding.
          - The selected best beam is converted back to logits for compatibility
            with the original external interface.
        """
        labels_tgt = labels.permute(1, 0)

        if self.training:
            tgt_inputs = labels_tgt[:-1, :]
            tgt_masks, tgt_padding_masks = create_mask(tgt_inputs, labels.device, self.vocab.token2id['_PAD_'])
            out = self._forward(features, tgt_inputs, tgt_masks, tgt_padding_masks, return_attn=False)
            return out

        out_list = []
        attn_list = []
        pad_id = self.vocab.token2id['_PAD_']
        end_id = self.vocab.token2id['_END_']
        vocab_size = self.classifier[0].out_features

        for labels_i in range(len(labels)):
            feature_i = features[:, labels_i:labels_i + 1, :]

            # 1) Search best token sequence by beam search.
            #    best_tokens includes _START_.
            best_tokens = self._beam_search_decode_one(feature_i, labels.device)

            # 2) Scheme A compatibility:
            #    Old model.py expects logits and then applies argmax.
            #    Beam search has already selected the final token sequence, so we
            #    construct pseudo logits whose argmax exactly recovers best_tokens[1:].
            generated_tokens = best_tokens[1:, 0].long()  # remove _START_

            if generated_tokens.numel() == 0:
                generated_tokens = torch.tensor([end_id], dtype=torch.long, device=labels.device)

            outputs = torch.full(
                (generated_tokens.size(0), 1, vocab_size),
                fill_value=-1e9,
                dtype=torch.float,
                device=labels.device,
            )
            outputs.scatter_(
                dim=-1,
                index=generated_tokens.view(-1, 1, 1),
                src=torch.zeros((generated_tokens.size(0), 1, 1), dtype=torch.float, device=labels.device),
            )

            # 3) Optional attention visualization:
            #    replay the selected prefix only to collect cross-attention.
            #    Do not return these true logits, otherwise model.py's argmax would
            #    overwrite the beam-search result.
            if return_attn:
                if best_tokens.size(0) > 1:
                    decode_inputs = best_tokens[:-1, :]
                else:
                    decode_inputs = best_tokens

                tgt_masks, tgt_padding_masks = create_mask(decode_inputs, labels.device, pad_id)

                _, cross_attn = self._forward(
                    feature_i,
                    decode_inputs,
                    tgt_masks,
                    tgt_padding_masks,
                    return_attn=True
                )
                if cross_attn is not None:
                    attn_list.append(cross_attn[0])  # [nhead, T, S]
                else:
                    attn_list.append(None)

            out_list.append(outputs)

        if return_attn:
            return out_list, {"cross_attn_last": attn_list}

        return out_list
