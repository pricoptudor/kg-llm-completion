"""vLLM-backed scorer for KG completion — fast log-prob candidate scoring.

vLLM batches all (prefix + candidate) prompts with continuous batching + shared-
prefix caching, so it scores candidates far faster than the per-query HF loop.

- score_*_candidates(...) -> scores for a subset of entities (sampled eval).
- score_tails/score_heads(...) -> (batch, num_entities) over ALL entities, so this
  plugs into the FULL filtered harness `kg_llm.eval.ranking.evaluate` for the
  report-grade, SOTA-comparable numbers (rank gold among all 14,541).

Scoring reads vLLM's prompt_logprobs: send prefix+candidate as a prompt and sum the
log-probs vLLM assigns to the candidate's own tokens (length-normalized).
"""

from __future__ import annotations

import torch

from kg_llm.data.fb15k237 import FB15k237
from kg_llm.llm.sft_data import head_question, tail_question


class VLLMScorer:
    def __init__(
        self,
        llm,
        tokenizer,
        dataset: FB15k237,
        *,
        chat_template: bool = False,
        length_normalize: bool = True,
        lora_request=None,
    ) -> None:
        self.llm = llm
        self.tok = tokenizer
        self.ds = dataset
        self.chat_template = chat_template
        self.length_normalize = length_normalize
        self.lora_request = lora_request

        lead = "" if chat_template else " "
        self._cand_tokens = [
            self.tok(lead + dataset.entity_name(i), add_special_tokens=False).input_ids
            for i in range(dataset.num_entities)
        ]

    def _tail_prefix(self, head_id, relation_id) -> str:
        hn, rn = self.ds.entity_name(int(head_id)), self.ds.relation_name(int(relation_id))
        if self.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": tail_question(hn, rn)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        return f"Head: {hn}. Relation: {rn}. Tail:"

    def _head_prefix(self, relation_id, tail_id) -> str:
        tn, rn = self.ds.entity_name(int(tail_id)), self.ds.relation_name(int(relation_id))
        if self.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": head_question(tn, rn)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        return f"Tail: {tn}. Relation: {rn}. Head:"

    def _score(self, prefix: str, cand_ids) -> torch.Tensor:
        from vllm import SamplingParams

        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)

        prompts, ans_lens = [], []
        for i in cand_ids:
            ans = self._cand_tokens[int(i)]
            prompts.append({"prompt_token_ids": prefix_ids + ans})
            ans_lens.append(len(ans))

        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate(prompts, sp, lora_request=self.lora_request, use_tqdm=False)

        scores = torch.full((len(prompts),), float("-inf"))
        for j, (out, a) in enumerate(zip(outs, ans_lens)):
            if a == 0:
                continue
            pl = out.prompt_logprobs  # list[Optional[dict[int, Logprob]]]
            ids = prompts[j]["prompt_token_ids"]
            s = 0.0
            for pos in range(plen, plen + a):
                s += pl[pos][ids[pos]].logprob
            scores[j] = s / a if self.length_normalize else s
        return scores

    # subset scoring (sampled eval)
    def score_tail_candidates(self, head_id, relation_id, candidate_ids) -> torch.Tensor:
        return self._score(self._tail_prefix(head_id, relation_id), list(candidate_ids))

    def score_head_candidates(self, relation_id, tail_id, candidate_ids) -> torch.Tensor:
        return self._score(self._head_prefix(relation_id, tail_id), list(candidate_ids))

    # full-entity scoring (report-grade, plugs into kg_llm.eval.ranking.evaluate)
    @torch.no_grad()
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        allids = list(range(self.ds.num_entities))
        out = torch.empty(len(heads), self.ds.num_entities)
        for k in range(len(heads)):
            out[k] = self._score(self._tail_prefix(heads[k], relations[k]), allids)
        return out

    @torch.no_grad()
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        allids = list(range(self.ds.num_entities))
        out = torch.empty(len(relations), self.ds.num_entities)
        for k in range(len(relations)):
            out[k] = self._score(self._head_prefix(relations[k], tails[k]), allids)
        return out
