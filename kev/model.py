"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option), opt (per-token option index within its
    question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal)."""
    L = max(len(s) for s in segs)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer dimension (head capacity knob)."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False):
        super().__init__()
        # backbone only (no vocab head): we never generate text.
        # eager on MPS/CPU (known-good with our float 4D mask); SDPA on CUDA (accepts arbitrary additive masks).
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32, attn_implementation=attn).model
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.option_isolation = option_isolation
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in SPECIAL]}} if special_embeddings else {}
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation setting; use this from serving/eval code."""
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records."""
        L = max(len(e["ids"]) for e in encs)
        ids = torch.full((len(encs), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(encs), L), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, opts=[e["opt"] for e in encs] if isolate else None)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        return self._readout(self.hidden(enc), enc)

    def forward_batch(self, encs):
        """List (per record) of lists (per question) of logits, from one padded forward pass."""
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
