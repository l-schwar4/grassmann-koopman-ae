"""
Global Koopman autoencoder compatible with the current generic train.py.

Model
-----
    z_t          = E(x_t)
    z_hat_{t+1}  = K z_t
    x_hat_t      = D(z_t)
    x_hat_{t+1}  = D(z_hat_{t+1})

The default autonomous rollout is the classical global-Koopman rollout:
encode once, repeatedly apply K in latent space, and decode the requested
states. Decode/re-encode inference is selected on held-out trajectories by the trainer.

Compatibility with the current train.py
----------------------------------------
- constructor accepts YAML model kwargs directly
- exposes rollout_mode so train.py can pass rollout_mode
- rollout accepts rollout_mode
- loss accepts extra keyword arguments harmlessly
- optional Grassmann projection with a learned Koopman operator
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..utils import reencode_candidates
except ImportError:
    from utils import reencode_candidates


try:
    from .linear_decoder import build_linear_decoder
except ImportError:
    from model.linear_decoder import build_linear_decoder

try:
    from .koopman_grassmann_ae import (
        StateConditionedGrassmannRepresentation,
        state_conditioned_rollout_recurrence,
        state_conditioned_sequence_loss,
        state_space_mse,
    )
except ImportError:
    from model.koopman_grassmann_ae import (
        StateConditionedGrassmannRepresentation,
        state_conditioned_rollout_recurrence,
        state_conditioned_sequence_loss,
        state_space_mse,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def activation_from_name(name: str) -> type[nn.Module]:
    table: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[key]


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: type[nn.Module] = nn.SiLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden_dims:
        h = int(h)
        if h <= 0:
            raise ValueError("hidden layer widths must be positive")
        layers.extend((nn.Linear(prev, h), activation()))
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = h
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class KoopmanAutoencoder(StateConditionedGrassmannRepresentation, nn.Module):
    """Global latent-linear Koopman autoencoder."""

    def __init__(
        self,
        *,
        state_dim: int,
        latent_dim: int = 16,
        grassmann: bool = False,
        grassmann_subspace_dim: int = 4,
        subspace_selector_network_hidden_dims: Optional[Sequence[int]] = None,
        subspace_selector_network_activation: Optional[str] = None,
        grassmann_parameterization: str = "householder",
        grassmann_num_reflections: Optional[int] = None,
        encoder_hidden_dims: Sequence[int] = (64, 64),
        activation: str = "silu",
        dropout: float = 0.0,
        operator_init_scale: float = 0.99,
        operator_init_noise: float = 0.01,
        # Current train.py reads this attribute and forwards it to rollout().
        # "latent" is the training rollout; "reencode" is an inference option.
        rollout_mode: str = "latent",
        reencode_every: int = 1,
        subspace_update_every: int = -1,
        **legacy_options: object,
    ) -> None:
        super().__init__()

        legacy_selector_dims = legacy_options.pop("subspace_hidden_dims", None)
        if subspace_selector_network_hidden_dims is None:
            subspace_selector_network_hidden_dims = (
                legacy_selector_dims
                if legacy_selector_dims is not None else (32,)
            )
        elif legacy_selector_dims is not None:
            raise ValueError(
                "Use only subspace_selector_network_hidden_dims; "
                "subspace_hidden_dims is a legacy checkpoint option"
            )

        self._state_dim = int(state_dim)
        self._latent_dim = int(latent_dim)
        self.encoder_hidden_dims = tuple(int(v) for v in encoder_hidden_dims)
        self.activation_name = str(activation).lower()
        self.subspace_selector_network_activation_name = str(
            activation if subspace_selector_network_activation is None
            else subspace_selector_network_activation
        ).lower()
        self.dropout = float(dropout)
        self.operator_init_scale = float(operator_init_scale)
        self.operator_init_noise = float(operator_init_noise)
        self.rollout_mode = str(rollout_mode).lower()
        self.reencode_candidates = reencode_candidates(reencode_every)
        self.reencode_every = self.reencode_candidates[0]
        self.subspace_update_every = int(subspace_update_every)

        if self._state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self._latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.dropout < 0:
            raise ValueError("dropout must be nonnegative")
        if self.rollout_mode not in {"latent", "reencode"}:
            raise ValueError(
                "rollout_mode must be 'latent' or 'reencode'"
            )

        if self.reencode_every < 1:
            raise ValueError("reencode_every must be at least 1")
        if self.subspace_update_every == 0 or self.subspace_update_every < -1:
            raise ValueError(
                "subspace_update_every must be -1 or a positive integer"
            )
        act = activation_from_name(self.activation_name)
        selector_act = activation_from_name(
            self.subspace_selector_network_activation_name
        )

        encoder_body, encoder_output_dim = self._configure_grassmann(
            grassmann=grassmann,
            latent_dim=self._latent_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            grassmann_subspace_dim=grassmann_subspace_dim,
            subspace_selector_network_hidden_dims=(
                subspace_selector_network_hidden_dims
            ),
            grassmann_parameterization=grassmann_parameterization,
            grassmann_num_reflections=grassmann_num_reflections,
            activation_factory=selector_act,
            dropout=self.dropout,
        )

        self.encoder = build_mlp(
            self._state_dim,
            encoder_body,
            encoder_output_dim,
            activation=act,
            dropout=self.dropout,
        )
        self.decoder = build_linear_decoder(self._latent_dim, self._state_dim)

        # A distinct layer makes the E -> K -> D structure explicit and allows
        # hooks or a separate optimizer to target only the Koopman operator.
        operator_dim = self._latent_dim
        initial_K = self.operator_init_scale * torch.eye(operator_dim)
        initial_K += self.operator_init_noise * torch.randn_like(initial_K)
        self.koopman_operator = nn.Linear(
            operator_dim, operator_dim, bias=False
        )
        with torch.no_grad():
            self.koopman_operator.weight.copy_(initial_K)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def K(self) -> torch.Tensor:
        """Koopman matrix compatibility alias for diagnostics and callers."""
        return self.koopman_operator.weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_representation(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def apply_operator(self, z: torch.Tensor, K: Optional[torch.Tensor] = None,
                       effective_K: Optional[torch.Tensor] = None) -> torch.Tensor:
        operator = self.K if K is None else K
        return self.apply_batched_operator(z, operator if effective_K is None else effective_K)


    def forward(
        self,
        x_t: torch.Tensor,
        x_tp1: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.grassmann:
            z_t, _, psi_t = self.project_state(x_t)
            z_hat_tp1 = self.apply_operator(z_t)
            out = {
                "x_t": x_t, "z_t": z_t, "z_ambient_t": psi_t,
                "x_hat_t": self.decode(z_t), "z_hat_tp1": z_hat_tp1,
                "x_hat_tp1": self.decode(z_hat_tp1),
            }
            if x_tp1 is not None:
                psi_tp1 = self.ambient_dictionary(x_tp1)
                z_tp1 = self.project_dictionary(psi_tp1, x=x_tp1)
                out.update(x_tp1=x_tp1, z_tp1=z_tp1, z_ambient_tp1=psi_tp1)
            return out
        z_t = self.encode(x_t)
        x_hat_t = self.decode(z_t)
        z_hat_tp1 = self.apply_operator(z_t)
        x_hat_tp1 = self.decode(z_hat_tp1)

        out: dict[str, torch.Tensor] = {
            "x_t": x_t,
            "z_t": z_t,
            # Alias useful for generic latent-space diagnostics.
            "z_ambient_t": z_t,
            "x_hat_t": x_hat_t,
            "z_hat_tp1": z_hat_tp1,
            "x_hat_tp1": x_hat_tp1,
        }

        if x_tp1 is not None:
            psi_tp1 = self.ambient_dictionary(x_tp1)
            z_tp1 = self.project_dictionary(psi_tp1)
            out["x_tp1"] = x_tp1
            out["z_tp1"] = z_tp1
            out["z_ambient_tp1"] = z_tp1

        return out

    # ------------------------------------------------------------------
    # Autonomous rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        x0: torch.Tensor,
        steps: int,
        *,
        include_initial: bool = False,
        rollout_mode: Optional[str] = None,
        reencode_every: Optional[int] = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """
        Autonomous global-KAE rollout.

        Modes
        -----
        latent:
            z_0 = E(x_0), z_{t+1}=K z_t.  No re-encoding.
            This is the standard global Koopman-AE rollout.

        reencode:
            after every reencode_every-th prediction, set z=E(x) before the
            next K step.  reencode_every=1 re-encodes after every prediction.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")

        mode = self.rollout_mode if rollout_mode is None else str(rollout_mode).lower()
        interval = self.reencode_every if reencode_every is None else int(reencode_every)
        if interval < 1:
            raise ValueError("reencode_every must be at least 1")
        if mode not in {"latent", "reencode"}:
            raise ValueError(
                f"Unsupported rollout_mode={mode!r}; expected 'latent' or 'reencode'"
            )

        if self.grassmann:
            z0 = self.encode(x0)
            z_roll, x_roll = state_conditioned_rollout_recurrence(
                self, z0, int(steps), operator=self.K,
                rollout_mode=mode, reencode_every=interval,
                subspace_update_every=self.subspace_update_every,
            )
            if include_initial:
                z_roll = torch.cat((z0.unsqueeze(1), z_roll), dim=1)
                x_roll = torch.cat((x0.unsqueeze(1), x_roll), dim=1)
            return {
                "x_rollout": x_roll,
                "z_rollout": z_roll,
                "z_ambient_rollout": z_roll,
            }

        z = self.encode(x0)
        x = x0

        xs: list[torch.Tensor] = []
        zs: list[torch.Tensor] = []

        if include_initial:
            xs.append(x)
            zs.append(z)

        for _step in range(int(steps)):
            z_pred = self.apply_operator(z)
            x_pred = self.decode(z_pred)

            xs.append(x_pred)
            zs.append(z_pred)

            x = x_pred
            should_reencode = (
                mode == "reencode"
                and (_step + 1) % interval == 0
                and (_step + 1) < int(steps)
            )
            z = self.encode(x_pred) if should_reencode else z_pred

        z_roll = torch.stack(zs, dim=1)
        return {
            "x_rollout": torch.stack(xs, dim=1),
            "z_rollout": z_roll,
            "z_ambient_rollout": z_roll,
        }


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Lusch sequence losses
# -----------------------------------------------------------------------------


def _matrix_l2(model: nn.Module) -> torch.Tensor:
    """L2 penalty over matrix-valued trainable parameters (weights/operators)."""
    vals = [p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    if not vals:
        return next(model.parameters()).new_zeros(())
    return torch.stack(vals).sum()


def koopman_sequence_loss(
    model: KoopmanAutoencoder,
    sequence: torch.Tensor,
    *,
    # Lusch et al. Eq. (11)-(15)
    lusch_alpha1: float = 1.0,
    lusch_linf_weight: float = 0.0,
    lusch_l2_weight: float = 0.0,
    prediction_steps: Optional[int] = None,
    linearity_steps: Optional[int] = None,
    temporal_consistency_weight: float = 0.0,
    state_mse_group_sizes: Optional[Sequence[int]] = None,
    rollout_mode: Optional[str] = None,
    **_: object,
) -> dict[str, torch.Tensor]:
    """
    Sequence objective using the published Lusch losses. The prediction
    recurrence follows rollout_mode: latent applies only K, while reencode
    periodically applies E(D(z)).

    Lusch:
      L = alpha1 * (L_recon + L_pred) + L_lin
          + alpha2 * L_inf + alpha3 * ||W||_2^2

    The latent target in L_lin is NOT detached: Lusch trains the encoder and K
    jointly to make encoded future states agree with K^h E(x_0).
    """
    if model.grassmann:
        return state_conditioned_sequence_loss(
            model, sequence, lusch_alpha1=lusch_alpha1,
            lusch_linf_weight=lusch_linf_weight, lusch_l2_weight=lusch_l2_weight,
            prediction_steps=prediction_steps, linearity_steps=linearity_steps,
            temporal_consistency_weight=temporal_consistency_weight,
            state_mse_group_sizes=state_mse_group_sizes,
            rollout_mode=rollout_mode,
        )
    if sequence.ndim != 3 or sequence.shape[1] < 2:
        raise ValueError("sequence must have shape (B,T,state_dim) with T>=2")

    B, T, D = sequence.shape
    xflat = sequence.reshape(B * T, D)
    psi_flat = model.ambient_dictionary(xflat)
    z_flat = model.project_dictionary(psi_flat)
    z_obs = z_flat.reshape(B, T, model.latent_dim)
    batch_K = model.K

    alpha1 = float(lusch_alpha1)
    linf_weight = float(lusch_linf_weight)
    zero = z_obs.new_zeros(())

    # Reconstruction is also needed by the optional L-infinity diagnostic.
    x0 = sequence[:, 0]
    z0 = z_obs[:, 0]
    if alpha1 != 0.0 or linf_weight != 0.0:
        x0_hat = model.decode(z0)
        l_recon = state_space_mse(x0_hat, x0, state_mse_group_sizes) if alpha1 != 0.0 else zero
    else:
        x0_hat = None
        l_recon = zero

    max_h = T - 1
    sp = max_h if prediction_steps is None else min(int(prediction_steps), max_h)
    sl = max_h if linearity_steps is None else min(int(linearity_steps), max_h)
    if sp < 1 or sl < 1:
        raise ValueError("prediction_steps and linearity_steps must be >=1")

    mode = model.rollout_mode if rollout_mode is None else str(rollout_mode).lower()
    if mode not in {"latent", "reencode"}:
        raise ValueError(f"Unsupported rollout_mode={mode!r}")
    interval = int(model.reencode_every)

    prediction_horizon = sp if alpha1 != 0.0 else (1 if linf_weight != 0.0 else 0)
    hmax = max(sl, prediction_horizon)
    latent_steps = []
    linearity_targets = []
    projected_decodes: dict[int, torch.Tensor] = {}
    z = z0
    for step in range(1, hmax + 1):
        z_pred_step = model.apply_operator(z, batch_K)
        latent_steps.append(z_pred_step)
        linearity_targets.append(z_obs[:, step])
        should_reencode = (
            mode == "reencode"
            and step % interval == 0
            and step < hmax
        )
        if should_reencode:
            x_projected = model.decode(z_pred_step)
            projected_decodes[step - 1] = x_projected
            z = model.encode(x_projected)
        else:
            z = z_pred_step
    z_pred = torch.stack(latent_steps, dim=1)

    l_lin = F.mse_loss(
        z_pred[:, :sl], torch.stack(linearity_targets[:sl], dim=1)
    )
    if alpha1 != 0.0:
        missing = [h for h in range(sp) if h not in projected_decodes]
        decoded_missing = (
            model.decode(
                z_pred[:, missing].reshape(B * len(missing), model.latent_dim),
            ).reshape(B, len(missing), D)
            if missing else None
        )
        missing_positions = {h: i for i, h in enumerate(missing)}
        decoded = torch.stack(
            [
                projected_decodes[h]
                if h in projected_decodes
                else decoded_missing[:, missing_positions[h]]
                for h in range(sp)
            ],
            dim=1,
        )
        l_pred = state_space_mse(decoded, sequence[:, 1 : sp + 1], state_mse_group_sizes)
        x1_pred = decoded[:, 0]
    elif linf_weight != 0.0:
        x1_pred = (
            projected_decodes[0]
            if 0 in projected_decodes else model.decode(
                z_pred[:, 0]
            )
        )
        l_pred = zero
    else:
        x1_pred = None
        l_pred = zero

    if linf_weight != 0.0:
        assert x0_hat is not None and x1_pred is not None
        l_inf = (
            (x0_hat - x0).abs().max()
            + (x1_pred - sequence[:, 1]).abs().max()
        )
    else:
        l_inf = zero
    l2 = _matrix_l2(model) if float(lusch_l2_weight) != 0.0 else zero

    total = (
        float(lusch_alpha1) * (l_recon + l_pred)
        + l_lin
        + float(lusch_linf_weight) * l_inf
        + float(lusch_l2_weight) * l2
    )
    terms = {
        "loss": total,
        "reconstruction": l_recon,
        "prediction": l_pred,
        "linearity": l_lin,
        "linf": l_inf,
        "l2": l2,
    }
    return terms
