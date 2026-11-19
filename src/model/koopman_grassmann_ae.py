"""State-conditioned Grassmann projection and GKAE sequence losses."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def state_space_mse(prediction, target, group_sizes=None):
    """Average group MSEs equally, independent of each group's channel count."""
    if group_sizes is None:
        return F.mse_loss(prediction, target)
    if (not group_sizes or any(type(size) is not int or size < 1 for size in group_sizes)
            or sum(group_sizes) != target.shape[-1]):
        raise ValueError("state_mse_group_sizes must be positive integers summing to the state dimension")
    predictions = prediction.split(group_sizes, dim=-1)
    targets = target.split(group_sizes, dim=-1)
    return torch.stack([F.mse_loss(p, t) for p, t in zip(predictions, targets)]).mean()


class StateConditionedGrassmannRepresentation:
    def _configure_grassmann(
        self, *, grassmann: bool, latent_dim: int,
        encoder_hidden_dims: Sequence[int],
        grassmann_subspace_dim: int = 4,
        subspace_selector_network_hidden_dims: Sequence[int] = (32,),
        grassmann_parameterization: str = "householder",
        grassmann_num_reflections: Optional[int] = None,
        activation_factory=None, dropout: float = 0.0, **_: object,
    ):
        self.grassmann = bool(grassmann)
        self.dictionary_dim = int(latent_dim)
        self.grassmann_rank = int(grassmann_subspace_dim)
        self.subspace_dim = self.grassmann_rank
        self.grassmann_parameterization = str(grassmann_parameterization).lower()
        self.grassmann_num_reflections = int(
            self.grassmann_rank if grassmann_num_reflections is None
            else grassmann_num_reflections
        )
        self.subspace_selector_network_hidden_dims = tuple(
            int(v) for v in subspace_selector_network_hidden_dims
        )
        if self.grassmann:
            if not 2 <= self.grassmann_rank < self.dictionary_dim:
                raise ValueError(
                    "grassmann=True requires 2 <= grassmann_subspace_dim < latent_dim"
                )
            if self.grassmann_parameterization not in {"householder", "qr"}:
                raise ValueError(
                    "grassmann_parameterization must be 'householder' or 'qr'"
                )
            if self.grassmann_num_reflections < 1:
                raise ValueError("grassmann_num_reflections must be positive")
            activation_factory = nn.SiLU if activation_factory is None else activation_factory
            layers = []
            width = self.state_dim
            for hidden in self.subspace_selector_network_hidden_dims:
                layers.extend((nn.Linear(width, hidden), activation_factory()))
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
                width = hidden
            output_width = (
                self.grassmann_num_reflections * (self.dictionary_dim - 1)
                if self.grassmann_parameterization == "householder"
                else self.dictionary_dim * self.grassmann_rank
            )
            output = nn.Linear(width, output_width)
            layers.append(output)
            self.subspace_net = nn.Sequential(*layers)
            with torch.no_grad():
                nn.init.normal_(output.weight, std=0.02)
                nn.init.normal_(output.bias, std=0.02)
        else:
            self.subspace_net = None
        return tuple(int(v) for v in encoder_hidden_dims), int(latent_dim)

    def predict_subspace(self, x: torch.Tensor) -> torch.Tensor:
        if not self.grassmann:
            raise RuntimeError("predict_subspace requires grassmann=True")
        if self.grassmann_parameterization == "householder":
            vectors = self.predict_reflection_vectors(x)
            return self.basis_from_context(vectors, dtype=x.dtype)
        shape = (*x.shape[:-1], self.dictionary_dim, self.grassmann_rank)
        raw = self.subspace_net(x).reshape(shape).float()
        q = torch.linalg.qr(raw, mode="reduced").Q
        if not torch.isfinite(q).all():
            raise FloatingPointError("Non-finite state-conditioned Grassmann QR basis")
        return q.to(dtype=x.dtype)

    def basis_from_context(
        self, context: torch.Tensor, *, dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Extract a basis from already-predicted vectors without rerunning G."""
        if self.grassmann_parameterization != "householder":
            return context if dtype is None else context.to(dtype)
        seed = torch.zeros(
            *context.shape[:-2], self.dictionary_dim, self.grassmann_rank,
            device=context.device, dtype=context.dtype,
        )
        diagonal = torch.arange(self.grassmann_rank, device=context.device)
        seed[..., diagonal, diagonal] = 1.0
        basis = self._apply_reflections(seed, context, forward=True)
        return basis if dtype is None else basis.to(dtype)

    def predict_reflection_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Return nonzero Householder normals shaped [..., reflections, D]."""
        if self.grassmann_parameterization != "householder":
            raise RuntimeError("reflection vectors require Householder parameterization")
        raw = self.subspace_net(x).reshape(
            *x.shape[:-1], self.grassmann_num_reflections,
            self.dictionary_dim - 1,
        )
        fixed = torch.ones(*raw.shape[:-1], 1, device=raw.device, dtype=raw.dtype)
        return torch.cat((raw, fixed), dim=-1)

    @staticmethod
    def _apply_reflections(
        values: torch.Tensor, vectors: torch.Tensor, *, forward: bool,
        inverse_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply H1..Hs (forward) or Hs..H1 without materializing H."""
        result = values
        indices = range(vectors.shape[-2]) if forward else range(vectors.shape[-2]-1, -1, -1)
        for index in indices:
            vector = vectors[..., index, :]
            inverse_norm = (
                vector.square().sum(-1, keepdim=True)
                .clamp_min(1.0e-12).reciprocal()
                if inverse_norms is None else inverse_norms[..., index, :]
            )
            if result.shape[-1] == vector.shape[-1]:
                inner = (result * vector).sum(-1, keepdim=True)
                result = result - 2.0 * vector * inner * inverse_norm
            else:
                inner = torch.einsum("...d,...dr->...r", vector, result)
                result = result - (
                    2.0 * vector.unsqueeze(-1) * inner.unsqueeze(-2)
                    * inverse_norm.unsqueeze(-1)
                )
        return result

    def project_with_reflections(
        self, psi: torch.Tensor, vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q M_r Q^T psi implicitly for row-oriented observations."""
        inverse_norms = vectors.square().sum(-1, keepdim=True).clamp_min(
            1.0e-12
        ).reciprocal()
        transformed = self._apply_reflections(
            psi, vectors, forward=False, inverse_norms=inverse_norms
        )
        masked = transformed.clone()
        masked[..., self.grassmann_rank:] = 0.0
        return self._apply_reflections(
            masked, vectors, forward=True, inverse_norms=inverse_norms
        ).to(psi.dtype)

    def project_at_state(self, x: torch.Tensor, psi: Optional[torch.Tensor] = None):
        psi = self.encoder(x) if psi is None else psi
        if self.grassmann_parameterization == "householder":
            vectors = self.predict_reflection_vectors(x)
            return self.project_with_reflections(psi, vectors), vectors, psi
        basis = self.predict_subspace(x)
        return self.project_with_basis(psi, basis), basis, psi


    def ambient_dictionary(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.encoder(x)

    def project_state(self, x: torch.Tensor, psi: Optional[torch.Tensor] = None):
        return self.project_at_state(x, psi)

    @staticmethod
    def project_with_basis(psi: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Project one or more dictionary states with one window-fixed basis."""
        coordinates = torch.einsum("...dr,...d->...r", basis, psi)
        return torch.einsum("...dr,...r->...d", basis, coordinates)

    def encode_training_window(self, sequence):
        """Select one subspace at the window start and hold it for all targets."""
        batch, time, state_dim = sequence.shape
        psi = self.ambient_dictionary(sequence.reshape(-1, state_dim)).reshape(
            batch, time, self.dictionary_dim,
        )
        if not self.grassmann:
            return psi, None
        _, context, _ = self.project_at_state(sequence[:, 0], psi[:, 0])
        return self.project_dictionary(psi, basis=context[:, None]), context

    def _encode_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_at_state(x)[0] if self.grassmann else self.encoder(x)

    def project_dictionary(self, psi: torch.Tensor, basis=None, x: Optional[torch.Tensor] = None):
        if not self.grassmann:
            return psi
        if basis is None:
            if x is None:
                raise ValueError("state x is required to predict a state-conditioned subspace")
            return self.project_at_state(x, psi)[0]
        if (self.grassmann_parameterization == "householder"
                and basis.shape[-2:] == (
                    self.grassmann_num_reflections, self.dictionary_dim
                )):
            return self.project_with_reflections(psi, basis)
        coords = torch.einsum("...dr,...d->...r", basis, psi)
        return torch.einsum("...dr,...r->...d", basis, coords)


    @staticmethod
    def apply_batched_operator(z, operator):
        return z @ operator.T if operator.ndim == 2 else torch.einsum("...ij,...j->...i", operator, z)


def state_conditioned_rollout_recurrence(
    model, initial, steps, *, operator, rollout_mode,
    reencode_every, subspace_update_every=-1,
):
    """Shared predicted-state recurrence for training and autonomous rollout."""
    mode = str(rollout_mode).lower()
    if mode not in {"latent", "reencode"}:
        raise ValueError(
            f"Unsupported rollout_mode={mode!r}; expected 'latent' or 'reencode'"
        )
    interval = int(reencode_every)
    if interval < 1:
        raise ValueError("reencode_every must be at least 1")
    subspace_interval = int(subspace_update_every)
    if subspace_interval == 0 or subspace_interval < -1:
        raise ValueError(
            "subspace_update_every must be -1 or a positive integer"
        )
    horizon = int(steps)
    if horizon < 1:
        raise ValueError("steps must be positive")

    if mode == "latent" and subspace_interval == -1:
        predictions = []
        current = initial
        for _ in range(horizon):
            current = model.apply_operator(current, operator, effective_K=operator)
            predictions.append(current)
        predicted = torch.stack(predictions, dim=1)
        return predicted, model.decoder(predicted)

    current = initial
    predictions = []
    decoded = []
    for step in range(1, horizon + 1):
        predicted = model.apply_operator(
            current, operator, effective_K=operator
        )
        predicted_state = model.decoder(predicted)
        predictions.append(predicted)
        decoded.append(predicted_state)

        should_reencode = (
            mode == "reencode"
            and step % interval == 0
            and step < horizon
        )
        should_update_subspace = (
            subspace_interval > 0
            and step % subspace_interval == 0
            and step < horizon
        )
        if should_reencode:
            current = model.encode(predicted_state)
        elif should_update_subspace:
            current = model.project_dictionary(
                predicted, x=predicted_state,
            )
        else:
            current = predicted

    return torch.stack(predictions, dim=1), torch.stack(decoded, dim=1)


def grassmann_temporal_tv(projectors: torch.Tensor) -> torch.Tensor:
    """Mean adjacent chordal distance, averaged over time and trajectory rows."""
    if projectors.ndim != 4 or projectors.shape[-1] != projectors.shape[-2]:
        raise ValueError(
            "projectors must have shape [trajectory, time, dimension, dimension]"
        )
    if projectors.shape[1] < 2:
        return projectors.sum() * 0.0
    # Keeping trajectory and time as separate axes is essential: flattening
    # them before differencing would introduce false trial-boundary penalties.
    consecutive_differences = projectors[:, 1:] - projectors[:, :-1]
    return torch.linalg.matrix_norm(
        consecutive_differences, ord="fro"
    ).mean() / (2.0 ** 0.5)


def state_conditioned_sequence_loss(model, sequence, *, lusch_alpha1=1.0,
        lusch_linf_weight=0.0, lusch_l2_weight=0.0, prediction_steps=None,
        linearity_steps=None,
        temporal_consistency_weight=0.0,
        state_mse_group_sizes=None,
        rollout_mode=None, **_):
    if sequence.ndim != 3 or sequence.shape[1] < 2:
        raise ValueError("sequence must have shape [batch,time,state]")
    batch, time, state_dim = sequence.shape
    flat_x = sequence.reshape(-1, state_dim)
    observed, _ = model.encode_training_window(sequence)
    sp = min(time-1, time-1 if prediction_steps is None else int(prediction_steps))
    sl = min(time-1, time-1 if linearity_steps is None else int(linearity_steps))
    operator = model.K
    horizon = max(sp, sl)
    mode = model.rollout_mode if rollout_mode is None else str(rollout_mode).lower()
    initial = observed[:, 0]
    predicted, states = state_conditioned_rollout_recurrence(
        model, initial, horizon, operator=operator,
        rollout_mode=mode, reencode_every=model.reencode_every,
        subspace_update_every=model.subspace_update_every,
    )
    reconstruction = state_space_mse(model.decoder(initial), sequence[:, 0], state_mse_group_sizes)
    prediction = state_space_mse(states[:, :sp], sequence[:, 1:sp+1], state_mse_group_sizes)
    linearity = F.mse_loss(predicted[:, :sl], observed[:, 1:sl+1])
    zero = reconstruction.new_zeros(())
    linf = ((model.decoder(initial)-sequence[:,0]).abs().max() +
            (states[:,0]-sequence[:,1]).abs().max()) if float(lusch_linf_weight) else zero
    matrix_l2 = torch.stack([p.square().sum() for p in model.parameters()
                             if p.requires_grad and p.ndim >= 2]).sum() if float(lusch_l2_weight) else zero
    # Ground-truth projector variation is a regularizer only: these projectors
    # never replace the initial projector or enter the latent prediction recurrence.
    # Use the complete window, independently of the legacy temporal horizon option.
    if float(temporal_consistency_weight):
        truth_bases = model.predict_subspace(flat_x).reshape(
            batch, time, model.dictionary_dim, model.grassmann_rank,
        )
        truth_projectors = truth_bases @ truth_bases.transpose(-2, -1)
        temporal_consistency = grassmann_temporal_tv(truth_projectors)
    else:
        temporal_consistency = zero
    total = (float(lusch_alpha1)*(reconstruction+prediction)+linearity+
             float(lusch_linf_weight)*linf+float(lusch_l2_weight)*matrix_l2+
             float(temporal_consistency_weight)*temporal_consistency)
    return {"loss":total,"reconstruction":reconstruction,"prediction":prediction,
            "linearity":linearity,"linf":linf,
            "l2":matrix_l2,
            "temporal_consistency":temporal_consistency}
