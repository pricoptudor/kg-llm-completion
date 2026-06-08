"""vLLM-backed scorer for KG completion — fast log-prob candidate scoring.

vLLM batches all (prefix + candidate) prompts with continuous batching + shared-
prefix caching, so it scores candidates far faster than the per-query HF forward
loop. Same `score_*_candidates` interface as LLMScorer, so `evaluate_llm_sampled`
works unchanged, and it extends to full-candidate (report-grade) eval.

Scoring uses vLLM's prompt_logprobs: we send prefix+candidate as a prompt and read
the log-probabilities vLLM assigns to the candidate's own tokens, then sum / mean.

VALIDATION: on the Qwen3-1.7B SFT model (256-way sampled, n=1000) this should match
the HF LLMScorer result (MRR 0.418). That agreement cross-checks the vLLM path.
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
        outs = self.llm.generate(
            prompts, sp, lora_request=self.lora_request, use_tqdm=False
        )

        scores = torch.full((len(cand_ids),), float("-inf"))
        for j, (out, a) in enumerate(zip(outs, ans_lens)):
            if a == 0:
                continue
            pl = out.prompt_logprobs  # list[Optional[dict[int, Logprob]]], len = #prompt tokens
            ids = prompts[j]["prompt_token_ids"]
            s = 0.0
            for pos in range(plen, plen + a):
                s += pl[pos][ids[pos]].logprob
            scores[j] = s / a if self.length_normalize else s
        return scores

    def score_tail_candidates(self, head_id, relation_id, candidate_ids) -> torch.Tensor:
        return self._score(self._tail_prefix(head_id, relation_id), list(candidate_ids))

    def score_head_candidates(self, relation_id, tail_id, candidate_ids) -> torch.Tensor:
        return self._score(self._head_prefix(relation_id, tail_id), list(candidate_ids))
