"""LLM scorer for KG completion via per-candidate log-probability.

Plugs a causal language model into the SAME `Scorer` protocol the KGE models use,
so the LLM is judged by the identical filtered-ranking harness.

For a tail query (h, r, ?) we build a plain completion prompt from readable names,
e.g. "Head: Albert Einstein. Relation: place of birth. Tail:", and score each
candidate entity `e` by the model's length-normalized log-probability of e's name
as the continuation:

    score(e) = (1/|tokens(e)|) * sum_i log p(token_i | prompt + earlier tokens)

Higher (closer to 0) = more plausible.

Scoring ALL 14,541 entities per query is ~14.5k short forwards per query, which is
infeasible on a T4 for more than a handful of queries. So in practice we use
`evaluate_llm_sampled`: rank the gold against a sampled set of negatives. The
LLMScorer exposes `score_*_candidates` to score an arbitrary entity-id subset.

Implementation notes:
- Candidate names are tokenized ONCE into a padded matrix at init.
- Scoring is vectorized per candidate batch (one forward, gather, masked sum); we
  never call .item() in the hot path (that forces a GPU->CPU sync per candidate).
- Prompt-KV caching (use_kv_cache=True): the prompt is identical across all
  candidates of a query, so we encode it ONCE, snapshot its DynamicCache, and reuse
  it (batch_repeat_interleave) for every candidate batch instead of re-encoding the
  prompt ~14.5k times. Same log-probs, ~10x fewer forwards on full-candidate eval.
  Falls back to the uncached path automatically if the cache API mismatches.
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from kg_llm.data.fb15k237 import FB15k237
from kg_llm.llm.sft_data import head_question, tail_question


def tail_prompt(head_name: str, relation_name: str) -> str:
    return f"Head: {head_name}. Relation: {relation_name}. Tail:"


def head_prompt(tail_name: str, relation_name: str) -> str:
    return f"Tail: {tail_name}. Relation: {relation_name}. Head:"


class LLMScorer:
    """Wrap a HF causal LM so it can score candidate entity names for a query."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset: FB15k237,
        *,
        length_normalize: bool = True,
        cand_batch_size: int = 128,
        chat_template: bool = False,
        use_kv_cache: bool = True,
        device=None,
    ) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.ds = dataset
        self.length_normalize = length_normalize
        self.cand_batch_size = cand_batch_size
        # chat_template=True scores candidates as the assistant answer to a chat
        # prompt (for SFT/DPO models, matching how they were trained). False = the
        # plain-completion prompt (for the zero-shot base model).
        self.chat_template = chat_template
        self.use_kv_cache = use_kv_cache
        self._cache_warned = False
        self.device = device or next(model.parameters()).device

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # In chat mode the answer follows "...assistant\n" with no leading space;
        # in plain mode it follows "Tail:" so we prepend a space.
        lead = "" if chat_template else " "
        cand_tokens = [
            self.tok(lead + dataset.entity_name(i), add_special_tokens=False).input_ids
            for i in range(dataset.num_entities)
        ]
        E = dataset.num_entities
        lmax = max((len(t) for t in cand_tokens), default=1) or 1
        matrix = torch.zeros((E, lmax), dtype=torch.long)
        mask = torch.zeros((E, lmax))
        lengths = torch.zeros(E, dtype=torch.long)
        for i, t in enumerate(cand_tokens):
            a = len(t)
            if a:
                matrix[i, :a] = torch.tensor(t, dtype=torch.long)
                mask[i, :a] = 1.0
            lengths[i] = a
        self._cand_matrix = matrix.to(self.device)
        self._cand_mask = mask.to(self.device)
        self._cand_len = lengths.to(self.device)

    def _score_indices(self, prefix: str, idx: torch.Tensor) -> torch.Tensor:
        """Dispatch to the cached fast path, falling back to uncached on any error."""
        if self.use_kv_cache:
            try:
                return self._score_indices_cached(prefix, idx)
            except Exception as e:  # noqa: BLE001 - cache API drift -> safe fallback
                if not self._cache_warned:
                    print(f"[scorer] KV-cache path disabled ({type(e).__name__}: {e}); "
                          "using uncached scoring.")
                    self._cache_warned = True
                self.use_kv_cache = False
        return self._score_indices_nocache(prefix, idx)

    @torch.no_grad()
    def _score_indices_cached(self, prefix: str, idx: torch.Tensor) -> torch.Tensor:
        """Same scores as _score_indices_nocache, but encode the prompt ONCE and reuse
        its KV cache across candidate batches (the prompt is shared by all candidates)."""
        import copy

        dev = self.device
        idx = idx.to(dev)
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=dev)  # (1, plen)

        pout = self.model(input_ids=prefix_t, use_cache=True)
        prefix_cache = pout.past_key_values            # DynamicCache, batch 1, len plen
        last_logit = pout.logits[:, -1, :].float()     # (1, V): predicts candidate token 0

        n = idx.shape[0]
        out = torch.empty(n, device=dev)
        for start in range(0, n, self.cand_batch_size):
            chunk = idx[start : start + self.cand_batch_size]
            B = chunk.shape[0]
            lens = self._cand_len[chunk]
            max_a = int(lens.max())
            if max_a == 0:
                out[start : start + B] = float("-inf")
                continue
            cand_block = self._cand_matrix[chunk][:, :max_a]   # (B, max_a)
            mask_block = self._cand_mask[chunk][:, :max_a]     # (B, max_a)

            # Fresh expanded copy of the pristine prefix cache (the forward appends to it).
            cache = copy.deepcopy(prefix_cache)
            ret = cache.batch_repeat_interleave(B)             # 1 -> B identical rows
            if ret is not None:
                cache = ret
            attn = torch.cat([torch.ones(B, plen, device=dev), mask_block], dim=1).long()
            cache_pos = torch.arange(plen, plen + max_a, device=dev)

            logits_cand = self.model(
                input_ids=cand_block,
                attention_mask=attn,
                past_key_values=cache,
                use_cache=False,
                cache_position=cache_pos,
            ).logits.float()                                   # (B, max_a, V)

            # Align predictions to candidate tokens 0..max_a-1:
            #   token 0  <- cached prefix's last logit
            #   token j  <- logits_cand[:, j-1]
            pred = torch.cat(
                [last_logit.expand(B, -1).unsqueeze(1), logits_cand[:, : max_a - 1, :]],
                dim=1,
            )                                                  # (B, max_a, V)
            tok_lp = torch.log_softmax(pred, dim=-1).gather(2, cand_block.unsqueeze(-1)).squeeze(-1)
            summed = (tok_lp * mask_block).sum(dim=1)
            if self.length_normalize:
                summed = summed / lens.clamp(min=1).float()
            out[start : start + B] = summed
        return out.cpu()

    @torch.no_grad()
    def _score_indices_nocache(self, prefix: str, idx: torch.Tensor) -> torch.Tensor:
        """Score the entities in `idx` (1-D long) as continuations of `prefix`.

        Returns a (len(idx),) tensor aligned to `idx`.
        """
        dev = self.device
        idx = idx.to(dev)
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=dev)
        n = idx.shape[0]
        out = torch.empty(n, device=dev)

        for start in range(0, n, self.cand_batch_size):
            chunk = idx[start : start + self.cand_batch_size]
            B = chunk.shape[0]
            lens = self._cand_len[chunk]
            max_a = int(lens.max())
            if max_a == 0:
                out[start : start + B] = float("-inf")
                continue
            cand_block = self._cand_matrix[chunk][:, :max_a]
            mask_block = self._cand_mask[chunk][:, :max_a]

            input_ids = torch.cat([prefix_t.unsqueeze(0).expand(B, plen), cand_block], dim=1)
            attn = torch.cat([torch.ones(B, plen, device=dev), mask_block], dim=1).long()

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            sub_lp = torch.log_softmax(logits[:, plen - 1 : plen - 1 + max_a, :].float(), dim=-1)
            tok_lp = sub_lp.gather(2, cand_block.unsqueeze(-1)).squeeze(-1)
            summed = (tok_lp * mask_block).sum(dim=1)
            if self.length_normalize:
                summed = summed / lens.clamp(min=1).float()
            out[start : start + B] = summed

        return out.cpu()

    # candidate-subset scoring (used by evaluate_llm_sampled)
    def _tail_prefix(self, head_id, relation_id) -> str:
        hn, rn = self.ds.entity_name(int(head_id)), self.ds.relation_name(int(relation_id))
        if self.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": tail_question(hn, rn)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        return tail_prompt(hn, rn)

    def _head_prefix(self, relation_id, tail_id) -> str:
        tn, rn = self.ds.entity_name(int(tail_id)), self.ds.relation_name(int(relation_id))
        if self.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": head_question(tn, rn)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        return head_prompt(tn, rn)

    def score_tail_candidates(self, head_id, relation_id, candidate_ids) -> torch.Tensor:
        prefix = self._tail_prefix(head_id, relation_id)
        return self._score_indices(prefix, torch.as_tensor(candidate_ids, dtype=torch.long))

    def score_head_candidates(self, relation_id, tail_id, candidate_ids) -> torch.Tensor:
        prefix = self._head_prefix(relation_id, tail_id)
        return self._score_indices(prefix, torch.as_tensor(candidate_ids, dtype=torch.long))

    # full Scorer protocol (scores ALL entities — exact but expensive; small sets only)
    @torch.no_grad()
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        allidx = torch.arange(self.ds.num_entities)
        out = torch.empty(len(heads), self.ds.num_entities)
        for k in range(len(heads)):
            out[k] = self._score_indices(self._tail_prefix(heads[k], relations[k]), allidx)
        return out

    @torch.no_grad()
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        allidx = torch.arange(self.ds.num_entities)
        out = torch.empty(len(relations), self.ds.num_entities)
        for k in range(len(relations)):
            out[k] = self._score_indices(self._head_prefix(relations[k], tails[k]), allidx)
        return out


def _sample_negatives(num_entities: int, k: int, exclude: set[int], gen: torch.Generator) -> list[int]:
    """k distinct entity ids in [0, num_entities) not in `exclude`."""
    perm = torch.randperm(num_entities, generator=gen).tolist()
    out = []
    for x in perm:
        if x not in exclude:
            out.append(x)
            if len(out) == k:
                break
    return out


def _realistic_rank(scores: torch.Tensor) -> float:
    """Rank of the gold (index 0) among `scores`, with realistic tie handling."""
    gold = scores[0]
    others = scores[1:]
    greater = int((others > gold).sum())
    tied = int((others == gold).sum())
    return 1.0 + greater + 0.5 * tied


@torch.no_grad()
def evaluate_llm_sampled(
    scorer: LLMScorer,
    triples: torch.Tensor,
    filtered_index,
    num_entities: int,
    *,
    num_candidates: int = 256,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 3, 10),
):
    """Filtered ranking of the gold against `num_candidates-1` sampled negatives.

    Pools head and tail queries, like the full harness. Negatives exclude all
    known-true answers (the filtered protocol) and the gold. NOTE: metrics are over
    `num_candidates` candidates, NOT all 14,541 — comparable to KGE only if KGE is
    run under the same sampled protocol.
    """
    from kg_llm.eval.ranking import aggregate

    gen = torch.Generator().manual_seed(seed)
    k = num_candidates - 1
    ranks: list[float] = []

    for h, r, t in tqdm(triples.tolist(), desc="LLM eval", unit="triple"):
        excl = set(filtered_index.true_tails(h, r).tolist())
        excl.add(t)
        cand = torch.tensor([t] + _sample_negatives(num_entities, k, excl, gen), dtype=torch.long)
        ranks.append(_realistic_rank(scorer.score_tail_candidates(h, r, cand)))

        excl = set(filtered_index.true_heads(r, t).tolist())
        excl.add(h)
        cand = torch.tensor([h] + _sample_negatives(num_entities, k, excl, gen), dtype=torch.long)
        ranks.append(_realistic_rank(scorer.score_head_candidates(r, t, cand)))

    return aggregate(torch.tensor(ranks, dtype=torch.float), ks)
