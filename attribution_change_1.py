"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import logging
import time
import weakref
from functools import partial
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules


def gpu_mem_usage():
    """Log GPU 1 memory usage for debugging."""
    logger = logging.getLogger("attribution")
    
    if torch.cuda.is_available():
        gpu_id = 0
    
    try:
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
        max_memory = torch.cuda.max_memory_allocated(gpu_id) / 1024**3   # GB
        
        logger.info(
            f"GPU{gpu_id} Memory - Allocated: {memory_allocated:.2f}GB, "
            f"Reserved: {memory_reserved:.2f}GB, Max: {max_memory:.2f}GB"
        )
    except Exception as e:
        logger.info(f"Error getting GPU{gpu_id} memory info: {e}")

class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            activation_matrix, error_vectors, token_vectors, decoder_vecs, feature_output_hook
        )

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def _caching_hooks(self, feature_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            # output_vecs may be on CPU, so move to GPU for matmul
            output_vecs_device = output_vecs.to(grads.device)

            # The original einsum is memory-inefficient for feature hooks because
            # grads[read_index] creates a massive intermediate tensor when
            # read_index is a form of advanced indexing.
            # We now handle the simple case (error/token nodes) and the complex
            # case (feature nodes) separately.

            # Simple case for error/token nodes where read_index is a simple slice
            if isinstance(read_index, slice):
                proxy._batch_buffer[write_index] = einsum(
                    grads.to(output_vecs_device.dtype)[read_index],
                    output_vecs_device,
                    "batch position d_model, position d_model -> position batch",
                )
                return

            # Advanced case for feature nodes. We loop over positions to avoid OOM.
            if not isinstance(read_index, tuple) or len(read_index) != 2:
                raise TypeError(f"Unexpected read_index type for feature hook: {type(read_index)}")

            positions_for_activations = read_index[1]
            n_pos = grads.shape[1]
            result_buffer = torch.zeros_like(proxy._batch_buffer[write_index])

            for pos_idx in range(n_pos):
                activations_at_pos_mask = positions_for_activations == pos_idx
                if not torch.any(activations_at_pos_mask):
                    continue

                output_vecs_at_pos = output_vecs_device[activations_at_pos_mask]
                grads_at_pos = grads[:, pos_idx, :]  # Shape: (batch, d_model)

                # einsum("batch d_model, n_activations d_model -> n_activations batch")
                einsum_result = einsum(
                    grads_at_pos.to(output_vecs_device.dtype),
                    output_vecs_at_pos,
                    "b d, p d -> p b",
                )
                result_buffer[activations_at_pos_mask] = einsum_result

            proxy._batch_buffer[write_index] = result_buffer

        return hook_name, _hook_fn

    def _make_attribution_hooks(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """Create the complete backward-hook for computing attribution scores."""

        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        # Token-embedding nodes
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T





@torch.no_grad()
def select_scaled_decoder_vecs(
    activations: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    This function allocates the final tensor on the CPU to avoid GPU OOM errors
    when the number of active features is very large.
    """
    logger = logging.getLogger("attribution")

    total_activations = activations._nnz()
    if total_activations == 0:
        return torch.empty(
            0,
            transcoders[0].W_dec.shape[1],
            dtype=transcoders[0].W_dec.dtype,
            device="cpu",
        )

    d_model = transcoders[0].W_dec.shape[1]
    dtype = transcoders[0].W_dec.dtype

    logger.info("Allocating decoder vectors on CPU to save GPU memory.")
    # Allocate final tensor on CPU to avoid GPU OOM.
    final_tensor = torch.empty(total_activations, d_model, dtype=dtype, device="cpu")

    current_offset = 0
    for layer, row in enumerate(activations):
        gpu_mem_usage()
        logger.info(f"Processing layer {layer} for decoder vectors.")
        _, feat_idx = row.coalesce().indices()
        num_activations_in_layer = len(feat_idx)

        if num_activations_in_layer == 0:
            continue

        # Fetch decoder rows for the current layer on GPU
        decoder_rows = transcoders[layer].W_dec[feat_idx]

        # Copy the rows to the final CPU tensor
        final_tensor[
            current_offset : current_offset + num_activations_in_layer
        ] = decoder_rows.cpu()

        current_offset += num_activations_in_layer

        # Free GPU memory explicitly
        del decoder_rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Finished gathering all decoder vectors on CPU.")

    # The activation values are on the GPU, so move them to the CPU for the multiplication
    activation_vals_cpu = activations.values().cpu()
    final_tensor *= activation_vals_cpu[:, None]

    logging.getLogger("attribution").info(
        f"Final scaled decoder vectors created on CPU with shape: {final_tensor.shape}"
    )
    gpu_mem_usage()
    return final_tensor


@torch.no_grad()
def select_encoder_rows(
    activation_matrix: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return encoder rows for **active** features only.

    This function allocates the final tensor on the CPU to avoid GPU OOM errors
    when the number of active features is very large.
    """
    logger = logging.getLogger("attribution")

    total_activations = activation_matrix._nnz()
    if total_activations == 0:
        return torch.empty(
            0,
            transcoders[0].W_enc.shape[0],  # d_model
            dtype=transcoders[0].W_enc.dtype,
            device="cpu",
        )

    d_model = transcoders[0].W_enc.shape[0]
    dtype = transcoders[0].W_enc.dtype

    logger.info("Allocating encoder rows on CPU to save GPU memory.")
    # Allocate final tensor on CPU to avoid GPU OOM.
    final_tensor = torch.empty(total_activations, d_model, dtype=dtype, device="cpu")

    current_offset = 0
    for layer, row in enumerate(activation_matrix):
        logger.info(f"Processing layer {layer} for encoder rows.")
        _, feat_idx = row.coalesce().indices()
        num_activations_in_layer = len(feat_idx)

        if num_activations_in_layer == 0:
            continue

        # Fetch encoder rows for the current layer on GPU
        encoder_rows_gpu = transcoders[layer].W_enc.T[feat_idx]

        # Copy the rows to the final CPU tensor
        final_tensor[
            current_offset : current_offset + num_activations_in_layer
        ] = encoder_rows_gpu.cpu()

        current_offset += num_activations_in_layer

        # Free GPU memory explicitly
        del encoder_rows_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Finished gathering all encoder rows on CPU.")
    logging.getLogger("attribution").info(
        f"Final encoder rows tensor created on CPU with shape: {final_tensor.shape}"
    )
    return final_tensor


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    # This computation can be very memory-intensive with large edge matrices.
    # We force it to the CPU to prevent CUDA OOM errors, as the influence
    # calculation is intermediate and doesn't need to be on the GPU.
    # device = torch.device("cpu")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        logger.removeHandler(handler)

def generate_all_plots(activation_matrix, output_dir="all_plots_14"):
    """
    Generates a plot for each layer and token, saving them to a single directory.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    n_layers, n_pos, _ = activation_matrix.shape
    dense_activations = activation_matrix.cpu().to_dense()

    for layer_idx in range(n_layers):
        for pos_idx in range(n_pos):
            values_for_slice = dense_activations[layer_idx, pos_idx, :]
            sorted_activations, _ = torch.sort(values_for_slice, descending=True)

            if sorted_activations.numel() == 0:
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_activations.float().numpy())
            plt.yscale('symlog', linthresh=1e-5)
            plt.title(f'Layer {layer_idx}, Token {pos_idx} - Sorted Activations (symlog scale)')
            plt.xlabel('Feature Rank (sorted)')
            plt.ylabel('Activation Value (symlog scale)')
            plt.grid(True)

            save_path = os.path.join(output_dir, f"layer_{layer_idx}_token_{pos_idx}.png")
            plt.savefig(save_path)
            plt.close()
    
    return output_dir

def create_grouped_zips(plots_dir="all_plots_14"):
    """
    Groups plots from a directory into two zip files (one by layer, one by token),
    with plots organized into subdirectories within each zip.
    """
    import os
    import re
    import zipfile
    from collections import defaultdict

    layer_files = defaultdict(list)
    token_files = defaultdict(list)
    
    file_pattern = re.compile(r"layer_(\d+)_token_(\d+)\.png")

    for filename in os.listdir(plots_dir):
        match = file_pattern.match(filename)
        if match:
            layer_idx = int(match.group(1))
            token_idx = int(match.group(2))
            filepath = os.path.join(plots_dir, filename)
            layer_files[layer_idx].append(filepath)
            token_files[token_idx].append(filepath)

    # Create one zip file grouped by layer
    with zipfile.ZipFile("visualizations_by_layer_14.zip", 'w') as zipf:
        for layer_idx, files in layer_files.items():
            for file in files:
                arcname = os.path.join(f"layer_{layer_idx}", os.path.basename(file))
                zipf.write(file, arcname)
    
    # Create one zip file grouped by token
    with zipfile.ZipFile("visualizations_by_token_14.zip", 'w') as zipf:
        for token_idx, files in token_files.items():
            for file in files:
                arcname = os.path.join(f"token_{token_idx}", os.path.basename(file))
                zipf.write(file, arcname)


def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    update_interval=4,
    logger=None,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = ensure_tokenized(prompt, model.tokenizer)
    print("a")
    logits, activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )
    
    # Prune near-zero activations from the sparse matrix
    # epsilon = 1e-9  # Define your threshold for "very close to zero"
    # indices = activation_matrix.indices()
    # values = activation_matrix.values()
    
    # mask = torch.abs(values) > epsilon
    
    # activation_matrix = torch.sparse_coo_tensor(
    #     indices[:, mask], values[mask], activation_matrix.shape
    # )

    # plots_dir = generate_all_plots(activation_matrix)
    # create_grouped_zips(plots_dir)

    decoder_vecs = select_scaled_decoder_vecs(activation_matrix, model.transcoders)
    print("c")
    encoder_rows = select_encoder_rows(activation_matrix, model.transcoders)
    print("d")
    ctx = AttributionContext(
        activation_matrix, error_vecs, token_vecs, decoder_vecs, model.feature_output_hook
    )
    print("e")
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)
            print("attr")
            gpu_mem_usage()
            inject_values = encoder_rows[idx_batch].to(device)
            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=inject_values,
                retain_graph=n_visited < max_feature_nodes,
            )
            gpu_mem_usage()
            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph
