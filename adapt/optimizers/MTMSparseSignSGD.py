import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Iterable, Optional, Any


class MTMSparseSignSGD(Optimizer):
    """
    Multi-Template Sparse Sign SGD.

    Training loop
    ----------------------
    For each optimizer step:

        optimizer.begin_template_accumulation()

        for template_batch in template_batches:
            loss = model(template_batch)
            loss.backward()
            optimizer.accumulate_template()
            optimizer.zero_grad()

        optimizer.finalize()   # builds mask (if due) + sign (always)
        optimizer.step()       # applies sparse signed update

    Mask vs sign refresh
    --------------------
    - The MASK is static by default (computed once, reused forever) or
      refreshed every `mask_refresh` steps when `static_mask=False`.
    - The SIGN is always recomputed every step from the current templates,
      regardless of mask settings.

    Parameters
    ----------
    params          : iterable of parameters or param groups
    lr              : learning rate
    density_ratio   : fraction of parameters updated each step (sparsity)
    weight_decay    : decoupled weight decay applied only to masked coordinates
    static_mask     : if True the mask is computed once and never refreshed
    sparse_mode     : "topk" (deterministic) or "random" (stochastic baseline)
    mask_refresh    : refresh mask every N steps when static_mask=False
    dense           : if True skip sparsity and do a full signed-SGD step
    normalize_grads : normalize each template gradient to unit norm before
                      accumulation to prevent high-magnitude templates from
                      dominating the mask / sign statistics
    mask_method     : how to aggregate template gradients into a mask
                        "mean_abs"    - top-k by mean |g| across templates
                        "mean"        - top-k by |mean(g)| across templates
                        "variance"    - top-k by gradient variance
                        "vote"        - top-k by vote count (each template
                                        nominates its own top-k)
                        "intersection"- params that ALL templates agree on
                                        (falls back to "vote" if too few)
                        "union"       - params voted by ANY template, capped
                                        at k (falls back to "vote" if too few)
    sign_method     : how to aggregate template gradients into an update sign
                        "vote_sign"   - majority sign across template sign()
                        "mean_sign"   - sign of the mean gradient
    """

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float = 1e-3,
        density_ratio: float = 1e-3,
        weight_decay: float = 0.0,
        static_mask: bool = True,
        sparse_mode: str = "topk",
        mask_refresh: int = 1,
        dense: bool = False,
        normalize_grads: bool = True,
        mask_method: str = "mean_abs",
        sign_method: str = "vote_sign",
    ):
        if sparse_mode not in {"topk", "random"}:
            raise ValueError("sparse_mode must be 'topk' or 'random'")
        if mask_method not in {"mean_abs", "mean", "variance", "vote", "intersection", "union"}:
            raise ValueError(f"Unknown mask_method: {mask_method!r}")
        if sign_method not in {"vote_sign", "mean_sign", "mean"}:
            raise ValueError(f"Unknown sign_method: {sign_method!r}")

        defaults = dict(
            lr=lr,
            density_ratio=density_ratio,
            weight_decay=weight_decay,
            static_mask=static_mask,
            sparse_mode=sparse_mode,
            mask_refresh=mask_refresh,
            dense=dense,
        )

        super().__init__(params, defaults)

        self.mask_method = mask_method
        self.sign_method = sign_method
        self.normalize_grads = normalize_grads

        self._num_templates: int = 0
        self._param_needs_mask_refresh: set = set()

    # =========================================================
    # PRIVATE UTILITIES
    # =========================================================

    @staticmethod
    def _topk_indices(flat: Tensor, k: int) -> Tensor:
        """Return indices of the top-k absolute values in a 1-D tensor."""
        return torch.topk(flat.abs(), k).indices

    def _need_new_mask(self, p: Tensor, group: dict) -> bool:
        state = self.state[p]
        step = state.get("step_cnt", 0)
        return "mask" not in state or (
            not group["static_mask"] and step % group["mask_refresh"] == 0
        )

    def _k(self, p: Tensor, group: dict) -> int:
        return max(1, int(p.numel() * group["density_ratio"]))

    def _maybe_normalize(self, g: Tensor) -> Tensor:
        if not self.normalize_grads:
            return g
        return g / (g.norm() + 1e-8)

    # =========================================================
    # TEMPLATE ACCUMULATION CONTROL
    # =========================================================

    @torch.no_grad()
    def begin_template_accumulation(self) -> None:
        """
        Call once before the template gradient loop.

        Always clears sign accumulators (sign is recomputed every step).
        Also clears mask accumulators for parameters whose mask is due
        for a refresh.
        """
        self._num_templates = 0

        # Determine which params need a mask refresh this step.
        self._param_needs_mask_refresh = set()
        for group in self.param_groups:
            for p in group["params"]:
                if self._need_new_mask(p, group):
                    self._param_needs_mask_refresh.add(id(p))

        _mask_accum_keys = {"grad_sum", "grad_abs_sum", "grad_sq_sum", "vote"}
        _sign_accum_keys = {"sign_sum", "grad_sum_sign"}

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                pid = id(p)

                # Sign accumulators are always cleared — sign is always recomputed.
                for k in _sign_accum_keys:
                    state.pop(k, None)

                # Mask accumulators are only cleared when a refresh is due.
                if pid in self._param_needs_mask_refresh:
                    for k in _mask_accum_keys:
                        state.pop(k, None)

    # =========================================================
    # ACCUMULATE ONE TEMPLATE'S GRADIENTS
    # =========================================================

    @torch.no_grad()
    def accumulate_template(self) -> None:
        """
        Call after each template's backward pass (before zero_grad).

        Always accumulates sign statistics.
        Also accumulates mask statistics for parameters whose mask is
        due for a refresh.
        """
        self._num_templates += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = self._maybe_normalize(p.grad.detach())
                g_flat = g.view(-1)
                state = self.state[p]
                pk = self._k(p, group)

                # ---- mask statistics (only when refreshing) ----
                
                if id(p) in self._param_needs_mask_refresh:
                    if self.mask_method == "mean_abs":
                        state.setdefault("grad_abs_sum", torch.zeros_like(g))
                        state["grad_abs_sum"].add_(g.abs())

                    elif self.mask_method == "mean":
                        state.setdefault("grad_sum", torch.zeros_like(g))
                        state["grad_sum"].add_(g)

                    elif self.mask_method == "variance":
                        state.setdefault("grad_sum", torch.zeros_like(g))
                        state.setdefault("grad_sq_sum", torch.zeros_like(g))
                        state["grad_sum"].add_(g)
                        state["grad_sq_sum"].add_(g * g)

                    elif self.mask_method in {"vote", "union", "intersection"}:
                        state.setdefault(
                            "vote",
                            torch.zeros(p.numel(), device=p.device, dtype=torch.int32),
                        )
                        idx = self._topk_indices(g_flat, pk)
                        state["vote"].index_add_(
                            0, idx, torch.ones(idx.numel(), device=p.device, dtype=torch.int32)
                        )

                # ---- sign statistics  ----

                if self.sign_method == "vote_sign":
                    state.setdefault("sign_sum", torch.zeros_like(g))
                    state["sign_sum"].add_(g.sign())

                elif self.sign_method == "mean_sign" or self.sign_method == "mean":
                    state.setdefault("grad_sum_sign", torch.zeros_like(g))
                    state["grad_sum_sign"].add_(g)

    # =========================================================
    # FINALIZE: COMPUTE MASK (if due) + SIGN (always)
    # =========================================================

    @torch.no_grad()
    def finalize(self) -> None:
        """
        Call after all templates have been accumulated, before step().

        Always finalizes the sign from accumulated template statistics.
        Only rebuilds the mask for parameters whose mask is due for refresh.
        """
        K = max(1, self._num_templates)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                pk = self._k(p, group)

                # ---- rebuild mask (only when due) ----

                if id(p) in self._param_needs_mask_refresh:

                    if self.mask_method == "mean_abs":
                        imp = state["grad_abs_sum"] / K
                        state["mask"] = self._topk_indices(imp.view(-1), pk)

                    elif self.mask_method == "mean":
                        imp = state["grad_sum"] / K
                        state["mask"] = self._topk_indices(imp.view(-1), pk)

                    elif self.mask_method == "variance":
                        mean = state["grad_sum"] / K
                        var = state["grad_sq_sum"] / K - mean * mean
                        state["mask"] = self._topk_indices(var.view(-1), pk)

                    elif self.mask_method == "vote":
                        state["mask"] = torch.topk(state["vote"], pk).indices

                    elif self.mask_method == "intersection":
                        idx = torch.where(state["vote"] == K)[0]
                        state["mask"] = (
                            idx if idx.numel() >= 0#pk
                            else torch.topk(state["vote"], pk).indices
                        )

                    elif self.mask_method == "union":
                        idx = torch.where(state["vote"] > 0)[0]
                        state["mask"] = (
                            idx[:pk] if idx.numel() >= pk
                            else torch.topk(state["vote"], pk).indices
                        )

                # ---- rebuild sign ----

                if self.sign_method == "vote_sign":
                    state["sign"] = state["sign_sum"].sign()

                elif self.sign_method == "mean_sign":
                    state["sign"] = state["grad_sum_sign"].sign()

                elif self.sign_method == "mean":
                    state["sign"] = state["grad_sum_sign"] / K

                assert "sign" in state, (
                    f"sign was not set for param (shape={p.shape}). "
                    "Did you call accumulate_template() at least once?"
                )
                assert "mask" in state, (
                    f"mask was not set for param (shape={p.shape}). "
                    "Did you call accumulate_template() at least once?"
                )

    # =========================================================
    # STEP — APPLIES THE SPARSE SIGNED UPDATE
    # =========================================================

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None):
        """
        Apply the sparse signed gradient update.
        The mask and sign stored in state are used; p.grad is only
        consulted for the weight-decay term and the dense fallback.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            dense = group["dense"]

            for p in group["params"]:
                state = self.state[p]
                state["step_cnt"] = state.get("step_cnt", 0) + 1

                p_flat = p.view(-1)

                # ---- dense signed-SGD fallback ----
                if dense:
                    if p.grad is None:
                        continue
                    update = p.grad.sign()
                    if wd:
                        update = update.add(p, alpha=wd)
                    p.add_(update, alpha=-lr)
                    p.grad = None
                    continue

                # ---- sparse path ----
                assert "mask" in state, (
                    "No mask found. Call begin_template_accumulation() → "
                    "accumulate_template() → finalize() before step()."
                )
                assert "sign" in state, (
                    "No sign found. Call finalize() before step()."
                )

                mask: Tensor = state["mask"]
                sign_flat: Tensor = state["sign"].view(-1)

                if group["sparse_mode"] == "random":
                    mask = torch.randperm(p.numel(), device=p.device)[:self._k(p, group)]

                g_vals = sign_flat.index_select(0, mask)
                if wd:
                    if p.grad is None:
                        raise RuntimeError(
                            "weight_decay > 0 requires p.grad at step() time. "
                            "Run a backward pass after finalize() before calling step()."
                        )
                    g_vals = g_vals.add(p_flat.index_select(0, mask), alpha=wd)

                p_flat.index_add_(0, mask, g_vals, alpha=-lr)
                p.grad = None

        return loss