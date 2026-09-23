"""CUDA graphs for serving the hybrid (Qwen3.5) backbones on CUDA.

A served request is one or two forward passes (the state prefix on a cache miss, then the question rows continuing the
cached state) over a few hundred tokens. At that size the GPU finishes its work long before Python finishes issuing it:
a pass launches ~2,000 kernels, and the flash-linear-attention wrapper around every DeltaNet layer costs ~1 ms of CPU.
Kev-4B and Kev-9B both took ~60 ms per pass on an H100 whether the state had 22 or 2,192 tokens, with the GPU busy for
~17-21 ms of it. Replaying a captured graph issues the same kernels in one call.

A graph has fixed shapes, so passes are padded to buckets (bucket(), at most 1/8 larger). Capturing one costs ~0.4 s
(an H100, Kev-4B: a warm-up pass, the capture and the instantiation of ~2,500 nodes), so the first pass of a new bucket
runs the same code eagerly from the same buffers and the bucket joins `pending`; capture_pending() captures them later
(kev.serve does it on a background thread after the response, one graph per turn of its model lock). A bucket seen once
costs nothing extra, the request that met it never waits for its capture, and a request arriving meanwhile waits for at
most one. The padding is masked exactly:
- the state pass is LEFT-padded. The DeltaNet layers see zeroed inputs at the pads (their padding mask), so keys, values
  and queries are zero and the recurrent state is still zero when the real tokens start; the causal convolution sees
  the same zeros its own padding supplies. The attention layers mask the pads as keys, and a pad query attends to itself
  so no row is fully masked (a NaN there would survive the zeroing: NaN * 0 = NaN). Real tokens keep their position ids.
- question rows are right-padded (after every real token, as the eager path already does); the cached state's keys are
  right-aligned in the bucket and the slots before them are masked.
So the results equal the eager passes up to floating-point reassociation (another chunking of the DeltaNet scan, other
GEMM shapes), not bit for bit.

Memory is bounded by construction: every graph reads and writes the same flat buffers (per attention layer a key and a
value buffer of GRAPH_TOKENS rows, per DeltaNet layer conv and recurrent states for GRAPH_ROWS rows, one hidden-state buffer),
viewed at the graph's shape, and all graphs share one memory pool. That is safe because replays run one at a time
(kev.serve holds a lock), each refills what it reads, and its results are copied out before the next replay.
"""
from collections import OrderedDict

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

# Limits of the graphed passes (not the model's context: that is kev.model.MAX_STATE / SERVE_MAX_STATE)
GRAPH_TOKENS = 32768   # rows x (state + row) tokens one graphed pass may hold in the attention buffers (about 1 GB on Kev-4B and Kev-9B)
GRAPH_STATE = 1024     # longest state bucket the state pass graphs. A longer state pass is compute-bound, and the padding plus
                       # the explicit mask (no causal flash attention) made a 2,200-token one slower as a graph (L40S, Kev-4B:
                       # 223 vs 208 ms per request), so it runs eagerly
GRAPH_ROW = 1024       # longest question-row bucket the row pass graphs
GRAPH_ROWS = 8         # question rows per graphed pass; more run as several replays
GRAPHS_KEPT = 128      # captured graphs kept, least recently used evicted


def bucket(n, steps=8, floor=16):
    """n rounded up to one of `steps` steps per power of two, at least `floor`. Token counts that set the pass's work use 8
    (the padding costs at most 1/8 of it); the cached-state length in a question pass only lengthens attention, so it uses
    4 (at most 1/4 more attention) and an application with a fixed question set needs few graphs."""
    step = max(floor, (1 << (n - 1).bit_length()) // steps)
    return -(-n // step) * step


class BufferKV(DynamicLayer):
    """An attention cache layer over preallocated [N, heads, T, dim] views: the first `filled` positions hold the cached
    state, a pass writes its keys/values after them instead of concatenating (no allocation, no Python state change,
    so one layer object serves the warm-up, the capture and every replay)."""

    def __init__(self, keys, values, filled):
        super().__init__()
        self.buffers, self.filled, self.is_initialized = (keys, values), filled, True
        self.keys, self.values = keys[..., :filled, :], values[..., :filled, :]

    def update(self, key_states, value_states, *args, **kwargs):
        end = self.filled + key_states.shape[-2]
        for buf, new in zip(self.buffers, (key_states, value_states)): buf[..., self.filled:end, :].copy_(new)
        return tuple(buf[..., :end, :] for buf in self.buffers)


def set_linear(layer, conv, recurrent, previous):
    """Point a DeltaNet cache layer at these conv/recurrent state tensors (transformers' LinearAttentionLayer fields)."""
    layer.conv_states[0], layer.recurrent_states[0] = conv, recurrent
    layer.is_conv_states_initialized[0] = layer.is_recurrent_states_initialized[0] = True
    layer.conv_kernel_size[0], layer.has_previous_state[0] = conv.shape[-1], previous
    layer.device, layer.dtype = conv.device, conv.dtype


def is_attention(layer):
    return isinstance(layer, DynamicLayer)


class CudaGraphs:
    def __init__(self, lm, pad_id):
        self.lm, self.pad_id = lm, pad_id
        self.device, self.dtype = next(lm.parameters()).device, next(lm.parameters()).dtype
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = OrderedDict()   # key -> (graph, input buffer)
        self.pending = {}             # key -> (pass body, input buffer): buckets that ran eagerly, not captured yet
        self.captures = 0
        # learn the cache layout (which layers are attention, state shapes and dtypes) from one eager pass
        probe = DynamicCache(config=lm.config)
        with torch.no_grad():
            lm(input_ids=torch.full((1, 16), pad_id, device=self.device), past_key_values=probe, use_cache=True)
        # BufferKV and set_linear rely on this layout: plain attention layers and single-state DeltaNet layers that update
        # their states in place. Anything else (another cache layer type, record_past, which assigns instead of copying)
        # would leave the buffers stale and move probabilities silently, so refuse it here.
        if not all(type(l) is DynamicLayer or (type(l) is LinearAttentionLayer and l.number_of_states == 1 and not l.record_past) for l in probe.layers):
            raise ValueError(f"CUDA graphs support attention and single-state DeltaNet cache layers only; got {sorted({type(l).__name__ for l in probe.layers})}")
        # per layer, two flat buffers and the per-row shape of what they hold: attention keys/values [heads, T, dim] with T
        # set per graph (GRAPH_TOKENS rows x positions in total), DeltaNet conv/recurrent states (GRAPH_ROWS rows)
        self.slots = []
        for layer in probe.layers:
            ts = (layer.keys, layer.values) if is_attention(layer) else (layer.conv_states[0], layer.recurrent_states[0])
            size = [GRAPH_TOKENS * t.shape[1] * t.shape[3] if is_attention(layer) else GRAPH_ROWS * t[0].numel() for t in ts]
            self.slots.append((is_attention(layer), [torch.zeros(n, dtype=t.dtype, device=self.device) for n, t in zip(size, ts)], [t.shape[1:] for t in ts]))
        self.hidden = torch.zeros(GRAPH_ROWS * GRAPH_ROW * lm.config.hidden_size, dtype=self.dtype, device=self.device)

    def _views(self, rows, length):
        """Per layer, its two buffers viewed for `rows` rows: attention [rows, heads, length, dim], DeltaNet [rows, *state]."""
        def view(flat, shape, attention):
            shape = (rows, shape[0], length, shape[2]) if attention else (rows, *shape)
            return flat[:torch.Size(shape).numel()].view(shape)
        return [tuple(view(f, s, attention) for f, s in zip(flats, shapes)) for attention, flats, shapes in self.slots]

    def _cache(self, views, filled, previous):
        """A DynamicCache over buffer views: attention layers hold `filled` cached positions, DeltaNet layers the states."""
        cache = DynamicCache(config=self.lm.config)
        for i, (layer, v) in enumerate(zip(cache.layers, views)):
            if is_attention(layer): cache.layers[i] = BufferKV(*v, filled)
            else: set_linear(layer, *v, previous)
        return cache

    def _mask(self, allow):
        """Additive attention mask [B, 1, Lq, Lk] from a boolean one [B, Lq, Lk], in the backbone's dtype."""
        return torch.zeros(allow.shape, dtype=self.dtype, device=self.device).masked_fill(~allow, torch.finfo(self.dtype).min)[:, None]

    def _forward(self, ids, pos, full_mask, linear_mask, cache):
        return self.lm(input_ids=ids, position_ids=pos, attention_mask={"full_attention": full_mask, "linear_attention": linear_mask},
                       past_key_values=cache, use_cache=True).last_hidden_state

    def _replay(self, key, body, rows, fill=lambda: None):
        """One pass for `key` on `rows` (token ids, positions and lengths, int64): the graph's replay, or body() run eagerly
        when the bucket has no graph yet (it then waits in `pending`). body(buf) runs the pass reading the uploaded rows
        from buf; fill() first writes the other buffers the pass reads. Every body for a key reads and writes the same
        buffer views, so the one kept in `pending` stands for all of them."""
        if key in self.graphs:
            self.graphs.move_to_end(key)
            graph, buf = self.graphs[key]
        else:
            graph, buf = None, self.pending.setdefault(key, (body, torch.zeros((len(rows), len(rows[0])), dtype=torch.long, device=self.device)))[1]
        fill()
        buf.copy_(torch.tensor(rows, dtype=torch.long), non_blocking=True)
        if graph is not None: graph.replay()
        else: body(buf)

    @torch.no_grad()
    def capture_pending(self, limit=None):
        """Capture graphs for up to `limit` (None = all) buckets that have run eagerly. The caller must keep other passes
        out (kev.serve holds its model lock, one capture at a time, so a request waits for at most one ~0.4 s capture). Each
        capture overwrites the shared buffers, which is fine between passes: a pass refills them. The warm-up pass runs on a
        side stream first: Triton autotuning and cuBLAS setup must not happen inside a capture. thread_local: CUDA calls of
        other threads (a request tokenizing, a model card read) cannot invalidate the capture."""
        for _ in range(len(self.pending) if limit is None else min(limit, len(self.pending))):
            key, (body, buf) = self.pending.popitem()
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream): body(buf)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"): body(buf)
            self.graphs[key] = (graph, buf); self.captures += 1
            while len(self.graphs) > GRAPHS_KEPT: self.graphs.popitem(last=False)

    @torch.no_grad()
    def prefix(self, ids, pos):
        """Run the state tokens; returns a DynamicCache equal to the eager prefix pass's, or None when the state is longer
        than GRAPH_STATE."""
        S = len(ids); Sb = bucket(S)
        if Sb > GRAPH_STATE: return None
        views = self._views(1, Sb)

        def body(buf):   # buf = [ids | positions | real length], left-padded
            pad = Sb - buf[:, 2 * Sb:]
            i = torch.arange(Sb, device=self.device)
            q, k = i[None, :, None], i[None, None, :]
            allow = ((k <= q) & (k >= pad[:, :, None])) | (k == q)
            self._forward(buf[:, :Sb], buf[:, Sb:2 * Sb], self._mask(allow), (i[None] >= pad).long(), self._cache(views, 0, False))

        self._replay(("state", Sb), body, [[self.pad_id] * (Sb - S) + list(ids) + [0] * (Sb - S) + list(pos) + [S]])
        out = DynamicCache(config=self.lm.config)   # a copy: the buffers belong to the next replay
        for layer, v in zip(out.layers, views):
            if is_attention(layer):
                layer.keys, layer.values = (t[..., Sb - S:, :].clone() for t in v)
                layer.dtype, layer.device, layer.is_initialized = self.dtype, self.device, True
            else:
                set_linear(layer, *(t.clone() for t in v), True)
        return out

    @torch.no_grad()
    def branches(self, rows, cache, prefix_len):
        """Hidden states [L_i, d] (float32) of question rows continuing a cached state of prefix_len tokens, or None when a
        row or the state is too long to graph: what DecisionModel._rows_hidden computes with a cache."""
        Lb, Pb = bucket(max(len(ids) for ids, _ in rows)), bucket(prefix_len, steps=4)
        group = min(GRAPH_ROWS, GRAPH_TOKENS // (Pb + Lb))
        if Lb > GRAPH_ROW or group < 1: return None
        out = []
        for start in range(0, len(rows), group):
            out += self._branch_pass(rows[start:start + group], cache, prefix_len, Lb, Pb)
        return out

    def _branch_pass(self, rows, cache, S, Lb, Pb):
        N, T = len(rows), Pb + Lb
        views = self._views(N, T)
        hidden = self.hidden[:N * Lb * self.lm.config.hidden_size].view(N, Lb, -1)

        def fill():   # the cached state, replicated per row; attention keys/values right-aligned before Pb
            for layer, v in zip(cache.layers, views):
                if is_attention(layer):
                    for buf, t in zip(v, (layer.keys, layer.values)): buf[..., Pb - S:Pb, :].copy_(t[..., t.shape[-2] - S:, :])
                else:
                    for buf, t in zip(v, (layer.conv_states[0], layer.recurrent_states[0])): buf.copy_(t)

        def body(buf):   # buf rows = [ids | positions | row length | state length]
            rowlen, plen = buf[:, 2 * Lb:2 * Lb + 1], buf[:, 2 * Lb + 1:]
            q = torch.arange(Lb, device=self.device)[None, :, None]
            k = torch.arange(T, device=self.device)[None, None, :]
            allow = ((k < Pb) & (k >= Pb - plen[:, :, None])) | ((k >= Pb) & (k - Pb <= q) & (k - Pb < rowlen[:, :, None])) | (k - Pb == q)
            linear = (torch.arange(Lb, device=self.device)[None] < rowlen).long()
            hidden.copy_(self._forward(buf[:, :Lb], buf[:, Lb:2 * Lb], self._mask(allow), linear, self._cache(views, Pb, True)))

        self._replay(("rows", N, Lb, Pb), body,
                     [list(ids) + [self.pad_id] * (Lb - len(ids)) + list(p) + [0] * (Lb - len(p)) + [len(ids), S] for ids, p in rows], fill)
        h = hidden.float()
        return [h[r, :len(ids)] for r, (ids, _) in enumerate(rows)]
