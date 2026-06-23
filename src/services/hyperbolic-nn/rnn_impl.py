import torch
from torch import nn

import util


def _new_matrix(rows: int, cols: int, dtype: torch.dtype, eye: bool) -> nn.Parameter:
    """Create a trainable matrix with shape [rows, cols]."""
    parameter = nn.Parameter(torch.empty(rows, cols, dtype=dtype))
    if eye:
        nn.init.eye_(parameter)
    else:
        nn.init.xavier_uniform_(parameter)
    return parameter


def _new_bias(width: int, dtype: torch.dtype) -> nn.Parameter:
    """Create a zero bias with shape [1, width]."""
    return nn.Parameter(torch.zeros(1, width, dtype=dtype))


class EuclRNN(nn.Module):
    """Euclidean RNN cell.

    Args:
        input_dim: Input tensor width.
        num_units: Hidden state width.
        dtype: Parameter dtype.

    Inputs:
        inputs: Tensor with shape [batch, input_dim].
        state: Tensor with shape [batch, num_units].

    Returns:
        Tensor with shape [batch, num_units].
    """

    def __init__(self, input_dim: int, num_units: int, dtype: torch.dtype) -> None:
        """Initialize parameters for inputs [batch, input_dim] and states [batch, num_units]."""
        super().__init__()
        self.num_units = num_units
        self.W = _new_matrix(num_units, num_units, dtype, eye=False)
        self.U = _new_matrix(input_dim, num_units, dtype, eye=False)
        self.b = _new_bias(num_units, dtype)
        self.eucl_params = [self.W, self.U, self.b]
        self.hyp_params = []

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Run one recurrent step for tensors [batch, dim]."""
        return torch.tanh(
            torch.matmul(state, self.W) + torch.matmul(inputs, self.U) + self.b
        )


class EuclGRU(nn.Module):
    """Euclidean GRU cell.

    Args:
        input_dim: Input tensor width.
        num_units: Hidden state width.
        dtype: Parameter dtype.

    Inputs:
        inputs: Tensor with shape [batch, input_dim].
        state: Tensor with shape [batch, num_units].

    Returns:
        Tensor with shape [batch, num_units].
    """

    def __init__(self, input_dim: int, num_units: int, dtype: torch.dtype) -> None:
        """Initialize GRU gate parameters for inputs [batch, input_dim]."""
        super().__init__()
        self.num_units = num_units
        self.Wz = _new_matrix(num_units, num_units, dtype, eye=False)
        self.Uz = _new_matrix(input_dim, num_units, dtype, eye=False)
        self.bz = _new_bias(num_units, dtype)
        self.Wr = _new_matrix(num_units, num_units, dtype, eye=False)
        self.Ur = _new_matrix(input_dim, num_units, dtype, eye=False)
        self.br = _new_bias(num_units, dtype)
        self.Wh = _new_matrix(num_units, num_units, dtype, eye=False)
        self.Uh = _new_matrix(input_dim, num_units, dtype, eye=False)
        self.bh = _new_bias(num_units, dtype)
        self.eucl_params = [
            self.Wz,
            self.Uz,
            self.bz,
            self.Wr,
            self.Ur,
            self.br,
            self.Wh,
            self.Uh,
            self.bh,
        ]
        self.hyp_params = []

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Run one GRU step for tensors [batch, dim]."""
        z = torch.sigmoid(
            torch.matmul(state, self.Wz) + torch.matmul(inputs, self.Uz) + self.bz
        )
        r = torch.sigmoid(
            torch.matmul(state, self.Wr) + torch.matmul(inputs, self.Ur) + self.br
        )
        h_tilde = torch.tanh(
            torch.matmul(r * state, self.Wh) + torch.matmul(inputs, self.Uh) + self.bh
        )
        return (1.0 - z) * state + z * h_tilde


class HypRNN(nn.Module):
    """Hyperbolic RNN cell using Mobius affine transforms.

    Args:
        input_dim: Input tensor width.
        num_units: Hidden state width.
        inputs_geom: Whether inputs are `eucl` or `hyp`.
        bias_geom: Whether biases are `eucl` or `hyp`.
        c_val: Positive curvature scalar.
        non_lin: Output nonlinearity name.
        fix_biases: Whether bias parameters are frozen.
        fix_matrices: Whether matrix parameters are frozen.
        matrices_init_eye: Whether matrices start as identity.
        dtype: Parameter dtype.

    Returns:
        Tensor with shape [batch, num_units].
    """

    def __init__(
        self,
        input_dim: int,
        num_units: int,
        inputs_geom: str,
        bias_geom: str,
        c_val: float,
        non_lin: str,
        fix_biases: bool,
        fix_matrices: bool,
        matrices_init_eye: bool,
        dtype: torch.dtype,
    ) -> None:
        """Initialize hyperbolic RNN parameters and optimizer groups."""
        super().__init__()
        self.num_units = num_units
        self.c_val = c_val
        self.non_lin = non_lin
        self.inputs_geom = inputs_geom
        self.bias_geom = bias_geom
        eye = matrices_init_eye or fix_matrices
        self.W = _new_matrix(num_units, num_units, dtype, eye=eye)
        self.U = _new_matrix(input_dim, num_units, dtype, eye=eye)
        self.b = _new_bias(num_units, dtype)
        self.W.requires_grad_(not fix_matrices)
        self.U.requires_grad_(not fix_matrices)
        self.b.requires_grad_(not fix_biases)
        self.eucl_params = []
        if not fix_matrices:
            self.eucl_params.extend([self.W, self.U])
        if not fix_biases and bias_geom == "eucl":
            self.eucl_params.append(self.b)
        self.hyp_params = []
        if not fix_biases and bias_geom == "hyp":
            self.hyp_params.append(self.b)

    def one_rnn_transform(
        self,
        W: torch.Tensor,
        h: torch.Tensor,
        U: torch.Tensor,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic Wh + Ux + b for tensors with shape [batch, dim]."""
        hyp_x = x
        if self.inputs_geom == "eucl":
            hyp_x = util.exp_map_zero(x, self.c_val)
        hyp_b = b
        if self.bias_geom == "eucl":
            hyp_b = util.exp_map_zero(b, self.c_val)
        W_otimes_h = util.mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.mob_mat_mul(U, hyp_x, self.c_val)
        Wh_plus_Ux = util.mob_add(W_otimes_h, U_otimes_x, self.c_val)
        return util.mob_add(Wh_plus_Ux, hyp_b, self.c_val)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Run one hyperbolic RNN step for tensors [batch, dim]."""
        new_h = self.one_rnn_transform(self.W, state, self.U, inputs, self.b)
        return util.hyp_non_lin(
            new_h, non_lin=self.non_lin, hyp_output=True, c=self.c_val
        )


class HypGRU(nn.Module):
    """Hyperbolic GRU cell using Mobius gates.

    Args mirror `HypRNN`. Inputs and output have shape [batch, dim].
    """

    def __init__(
        self,
        input_dim: int,
        num_units: int,
        inputs_geom: str,
        bias_geom: str,
        c_val: float,
        non_lin: str,
        fix_biases: bool,
        fix_matrices: bool,
        matrices_init_eye: bool,
        dtype: torch.dtype,
    ) -> None:
        """Initialize hyperbolic GRU gate parameters and optimizer groups."""
        super().__init__()
        self.num_units = num_units
        self.c_val = c_val
        self.non_lin = non_lin
        self.inputs_geom = inputs_geom
        self.bias_geom = bias_geom
        eye = matrices_init_eye or fix_matrices
        self.Wz = _new_matrix(num_units, num_units, dtype, eye=eye)
        self.Uz = _new_matrix(input_dim, num_units, dtype, eye=eye)
        self.bz = _new_bias(num_units, dtype)
        self.Wr = _new_matrix(num_units, num_units, dtype, eye=eye)
        self.Ur = _new_matrix(input_dim, num_units, dtype, eye=eye)
        self.br = _new_bias(num_units, dtype)
        self.Wh = _new_matrix(num_units, num_units, dtype, eye=eye)
        self.Uh = _new_matrix(input_dim, num_units, dtype, eye=eye)
        self.bh = _new_bias(num_units, dtype)
        self.eucl_params = []
        self.hyp_params = []
        matrix_params = [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]
        bias_params = [self.bz, self.br, self.bh]
        for parameter in matrix_params:
            parameter.requires_grad_(not fix_matrices)
            if not fix_matrices:
                self.eucl_params.append(parameter)
        for parameter in bias_params:
            parameter.requires_grad_(not fix_biases)
            if not fix_biases and bias_geom == "eucl":
                self.eucl_params.append(parameter)
            if not fix_biases and bias_geom == "hyp":
                self.hyp_params.append(parameter)

    def one_rnn_transform(
        self,
        W: torch.Tensor,
        h: torch.Tensor,
        U: torch.Tensor,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic Wh + Ux + b for tensors with shape [batch, dim]."""
        hyp_b = b
        if self.bias_geom == "eucl":
            hyp_b = util.exp_map_zero(b, self.c_val)
        W_otimes_h = util.mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.mob_mat_mul(U, x, self.c_val)
        Wh_plus_Ux = util.mob_add(W_otimes_h, U_otimes_x, self.c_val)
        return util.mob_add(Wh_plus_Ux, hyp_b, self.c_val)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Run one hyperbolic GRU step for tensors [batch, dim]."""
        hyp_x = inputs
        if self.inputs_geom == "eucl":
            hyp_x = util.exp_map_zero(inputs, self.c_val)
        z = util.hyp_non_lin(
            self.one_rnn_transform(self.Wz, state, self.Uz, hyp_x, self.bz),
            non_lin="sigmoid",
            hyp_output=False,
            c=self.c_val,
        )
        r = util.hyp_non_lin(
            self.one_rnn_transform(self.Wr, state, self.Ur, hyp_x, self.br),
            non_lin="sigmoid",
            hyp_output=False,
            c=self.c_val,
        )
        r_point_h = util.mob_pointwise_prod(state, r, self.c_val)
        h_tilde = util.hyp_non_lin(
            self.one_rnn_transform(self.Wh, r_point_h, self.Uh, hyp_x, self.bh),
            non_lin=self.non_lin,
            hyp_output=True,
            c=self.c_val,
        )
        minus_h_oplus_htilde = util.mob_add(-state, h_tilde, self.c_val)
        return util.mob_add(
            state,
            util.mob_pointwise_prod(minus_h_oplus_htilde, z, self.c_val),
            self.c_val,
        )
