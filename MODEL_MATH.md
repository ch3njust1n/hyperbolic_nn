<!-- Created: 2026-06-23 15:50 UTC. Purpose: Explain the implemented hyperbolic neural network model, math, derivations, and visual intuition. This is permanent project documentation. -->

# Hyperbolic Model Math

This document explains the model implemented in `src/services/hyperbolic-nn/`. The main training path is `hyp_rnn.py`, the recurrent cells are in `rnn_impl.py`, and the Poincare ball operations are in `util.py`.

## 1. Model Overview

The main model is a sentence-pair classifier. Each example contains two padded token sequences and a label:

$$
(s_1, s_2, y)
$$

Each sentence is encoded by an RNN or GRU into a vector. Depending on the config, the hidden states live in either Euclidean space or the Poincare ball. The two sentence vectors are combined by a feed-forward layer and classified by multinomial logistic regression (MLR).

```mermaid
flowchart LR
    tokenIds1["Sentence 1 token ids"] --> embed1["Shared embedding table"]
    tokenIds2["Sentence 2 token ids"] --> embed2["Shared embedding table"]
    embed1 --> encoder1["Encoder 1 RNN or GRU"]
    embed2 --> encoder2["Encoder 2 RNN or GRU"]
    encoder1 --> sent1["Sentence vector s1"]
    encoder2 --> sent2["Sentence vector s2"]
    sent1 --> dist["Distance feature"]
    sent2 --> dist
    sent1 --> ffnn["FFNN combine layer"]
    sent2 --> ffnn
    dist --> ffnn
    ffnn --> mlr["Euclidean or hyperbolic MLR"]
    mlr --> logits["Class logits"]
    logits --> loss["Cross entropy loss"]
```

Core implementation references:

- `HyperbolicRNNModel.forward` in `src/services/hyperbolic-nn/hyp_rnn.py`
- `HypRNN` and `HypGRU` in `src/services/hyperbolic-nn/rnn_impl.py`
- `mob_add`, `exp_map_zero`, `log_map_zero`, and `poinc_dist_sq` in `src/services/hyperbolic-nn/util.py`

## 2. Poincare Ball Geometry

The hyperbolic space is represented by the Poincare ball:

$$
\mathbb{D}_c^n = \{x \in \mathbb{R}^n : c\|x\|^2 < 1\}
$$

where `c > 0` is the absolute curvature scale. The ball radius is:

$$
R = \frac{1}{\sqrt{c}}
$$

The code keeps vectors inside the ball with `project_hyp_vecs`:

$$
x \leftarrow x \cdot \min\left(1, \frac{(1-\epsilon)/\sqrt{c}}{\|x\|}\right)
$$

```mermaid
flowchart TB
    center["Origin: nearly Euclidean geometry"]
    boundary["Boundary: exponentially more room"]
    tree["Hierarchical data fits naturally"]
    center -->|"small norms"| tree
    tree -->|"deeper nodes move outward"| boundary
```

The Poincare metric is conformal to the Euclidean metric:

$$
g_x^c = \lambda_x^2 g^E
$$

with conformal factor:

$$
\lambda_x^c = \frac{2}{1-c\|x\|^2}
$$

This appears in `lambda_x`. As points approach the boundary, the denominator shrinks, so the geometry stretches distances.

## 3. Mobius Addition

Euclidean vector addition does not preserve hyperbolic geometry. The code uses Mobius addition:

$$
u \oplus_c v =
\frac{
(1 + 2c\langle u,v\rangle + c\|v\|^2)u
+ (1 - c\|u\|^2)v
}{
1 + 2c\langle u,v\rangle + c^2\|u\|^2\|v\|^2
}
$$

This is implemented by `mob_add`.

Why this formula: in the Poincare ball, translations are isometries, not straight Euclidean shifts. Mobius addition is the gyrovector-space analogue of translation. Near the origin, when norms are small:

$$
c\|u\|^2 \approx 0,\quad c\|v\|^2 \approx 0,\quad 2c\langle u,v\rangle \approx 0
$$

so:

$$
u \oplus_c v \approx u + v
$$

That is why hyperbolic layers behave like Euclidean layers near zero but differ strongly near the boundary.

## 4. Distance

The model uses squared Poincare distance for sentence-pair features and regularization:

$$
d_c(u,v)^2 =
\left(
\frac{2}{\sqrt{c}}
\tanh^{-1}\left(\sqrt{c}\|-u \oplus_c v\|\right)
\right)^2
$$

This is implemented by `poinc_dist_sq`.

Derivation sketch:

1. Move `u` to the origin using the hyperbolic translation `-u \oplus_c v`.
2. Measure radial distance from the origin.
3. In the Poincare ball, radial distance from `0` to `z` is:

$$
d_c(0,z)=\frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|z\|)
$$

Substituting:

$$
z = -u \oplus_c v
$$

gives the implemented distance.

## 5. Exponential And Logarithmic Maps

The model frequently moves between the tangent space and the ball.

The tangent space at the origin behaves like ordinary Euclidean space. The exponential map sends a tangent vector into the ball:

$$
\exp_0^c(v)
=
\tanh(\sqrt{c}\|v\|)
\frac{v}{\sqrt{c}\|v\|}
$$

This is `exp_map_zero`.

The logarithmic map returns a ball point to the origin tangent space:

$$
\log_0^c(y)
=
\frac{1}{\sqrt{c}}
\tanh^{-1}(\sqrt{c}\|y\|)
\frac{y}{\|y\|}
$$

This is `log_map_zero`.

At a nonzero base point `x`, the exponential map is:

$$
\exp_x^c(v)
=
x \oplus_c
\left(
\tanh\left(
\frac{\sqrt{c}\lambda_x^c\|v\|}{2}
\right)
\frac{v}{\sqrt{c}\|v\|}
\right)
$$

This is `exp_map_x` and is used by Riemannian SGD.

The matching logarithmic map is:

$$
\log_x^c(y)
=
\frac{2}{\sqrt{c}\lambda_x^c}
\tanh^{-1}\left(\sqrt{c}\|-x \oplus_c y\|\right)
\frac{-x \oplus_c y}{\|-x \oplus_c y\|}
$$

This is `log_map_x`.

## 6. Mobius Matrix Multiplication

Euclidean layers compute:

$$
Wx
$$

For hyperbolic vectors, the code uses Mobius matrix multiplication:

$$
M \otimes_c x
=
\frac{1}{\sqrt{c}}
\tanh\left(
\frac{\|Mx\|}{\|x\|}
\tanh^{-1}(\sqrt{c}\|x\|)
\right)
\frac{Mx}{\|Mx\|}
$$

This is `mob_mat_mul`.

Derivation intuition:

1. Map `x` to a tangent representation with radial coordinate:

$$
\tanh^{-1}(\sqrt{c}\|x\|)
$$

2. Apply the Euclidean matrix direction `Mx`.
3. Map back into the ball with `tanh`.

For diagonal gates, `mob_pointwise_prod` uses the same idea with:

$$
Mx = u \odot x
$$

## 7. Hyperbolic RNN Cell

The Euclidean RNN update is:

$$
h_t = \phi(h_{t-1}W + x_tU + b)
$$

The hyperbolic version replaces affine operations with Mobius operations:

$$
\tilde{h}_t =
(W \otimes_c h_{t-1})
\oplus_c
(U \otimes_c x_t)
\oplus_c
b
$$

Then the nonlinearity is applied through the tangent space:

$$
h_t =
\exp_0^c\left(
\phi(\log_0^c(\tilde{h}_t))
\right)
$$

This matches `HypRNN.forward` and `hyp_non_lin`.

If inputs or biases are configured as Euclidean, the code maps them into the ball first:

$$
x_t^{hyp}=\exp_0^c(x_t), \quad b^{hyp}=\exp_0^c(b)
$$

## 8. Hyperbolic GRU Cell

The Euclidean GRU is:

$$
z_t = \sigma(h_{t-1}W_z + x_tU_z + b_z)
$$

$$
r_t = \sigma(h_{t-1}W_r + x_tU_r + b_r)
$$

$$
\tilde{h}_t =
\tanh((r_t \odot h_{t-1})W_h + x_tU_h + b_h)
$$

$$
h_t = (1-z_t)\odot h_{t-1} + z_t\odot \tilde{h}_t
$$

The hyperbolic GRU keeps gates in the tangent space but hidden states in the ball.

```mermaid
flowchart LR
    hPrev["Previous hyperbolic state h"]
    xTok["Token embedding x"]
    gateZ["Mobius affine then log0 sigmoid z"]
    gateR["Mobius affine then log0 sigmoid r"]
    reset["Mobius pointwise product r with h"]
    cand["Candidate hyperbolic state"]
    interp["Mobius interpolation"]
    hNext["Next hyperbolic state"]

    hPrev --> gateZ
    xTok --> gateZ
    hPrev --> gateR
    xTok --> gateR
    gateR --> reset
    hPrev --> reset
    reset --> cand
    xTok --> cand
    cand --> interp
    gateZ --> interp
    hPrev --> interp
    interp --> hNext
```

The update gate is:

$$
z_t =
\sigma\left(
\log_0^c(
(W_z \otimes_c h_{t-1})
\oplus_c
(U_z \otimes_c x_t)
\oplus_c
b_z
)
\right)
$$

The reset gate is:

$$
r_t =
\sigma\left(
\log_0^c(
(W_r \otimes_c h_{t-1})
\oplus_c
(U_r \otimes_c x_t)
\oplus_c
b_r
)
\right)
$$

The reset operation is a Mobius diagonal product:

$$
r_t \odot_c h_{t-1}
$$

The candidate hidden state is:

$$
\tilde{h}_t =
\exp_0^c
\left(
\phi\left(
\log_0^c(
(W_h \otimes_c (r_t \odot_c h_{t-1}))
\oplus_c
(U_h \otimes_c x_t)
\oplus_c
b_h
)
\right)
\right)
$$

The final update is a hyperbolic interpolation from `h` toward `h_tilde`:

$$
h_t =
h_{t-1}
\oplus_c
\left(
z_t \odot_c
(-h_{t-1} \oplus_c \tilde{h}_t)
\right)
$$

This matches `HypGRU.forward`.

## 9. Sentence Encoding

For each token index, `encode_sentence` updates only active batch items:

$$
m_{i,t} = \mathbb{1}[\ell_i > t]
$$

$$
h_{i,t} =
m_{i,t}\hat{h}_{i,t}
+ (1-m_{i,t})h_{i,t-1}
$$

This prevents padded tokens from changing finished sentence states.

The final sentence vectors are:

$$
s_1 = \operatorname{Encoder}_1(x_1,\ldots,x_m)
$$

$$
s_2 = \operatorname{Encoder}_2(x_1,\ldots,x_n)
$$

If sentence geometry is hyperbolic, `s_1` and `s_2` are points in `\mathbb{D}_c^d`.

## 10. Sentence-Pair FFNN Layer

The model computes a distance feature:

$$
\delta =
\begin{cases}
\|s_1-s_2\|^2, & \text{Euclidean sentence geometry} \\
d_c(s_1,s_2)^2, & \text{Hyperbolic sentence geometry}
\end{cases}
$$

For Euclidean FFNN geometry:

$$
o =
s_1W_1 + s_2W_2 + b
$$

If `additional_features == "dsq"`:

$$
o =
s_1W_1 + s_2W_2 + b + \delta b_d
$$

For hyperbolic FFNN geometry:

$$
o =
(W_1 \otimes_c s_1)
\oplus_c
(W_2 \otimes_c s_2)
\oplus_c
b
$$

With the distance feature:

$$
o =
o \oplus_c (\delta \otimes_c b_d)
$$

Then the configured nonlinearity is applied in Euclidean space or through:

$$
\exp_0^c(\phi(\log_0^c(o)))
$$

## 11. Euclidean MLR

For Euclidean MLR, each class has a normal vector `a_k` and point `p_k`. The logit is:

$$
\ell_k(x) =
\langle x - p_k, a_k\rangle
$$

This is a signed distance-like score to a Euclidean hyperplane.

## 12. Hyperbolic MLR

Hyperbolic MLR uses a point `p_k` in the ball and a tangent normal vector `a_k`.

The code first translates the input by `-p_k`:

$$
v_k = -p_k \oplus_c x
$$

Then it computes:

$$
\ell_k(x)
=
\frac{2}{\sqrt{c}}
\|a_k\|
\sinh^{-1}
\left(
\sqrt{c}
\lambda_{v_k}^c
\left\langle v_k, \frac{a_k}{\|a_k\|}\right\rangle
\right)
$$

This appears in both `HyperbolicRNNModel.forward` and `MnistSanityClassifier.forward`.

```mermaid
flowchart TB
    ball["Poincare ball"]
    pointP["Class point p"]
    tangentA["Normal vector a"]
    boundary["Decision boundary"]
    sampleX["Input x"]

    pointP -->|"anchors class hyperplane"| boundary
    tangentA -->|"orients hyperplane"| boundary
    sampleX -->|"translated by -p plus x"| boundary
    ball --> boundary
```

Intuition:

- `p_k` chooses where the class boundary is anchored.
- `a_k` chooses the normal direction.
- The conformal factor `lambda` accounts for local stretching in the Poincare ball.
- The `asinh` term converts the signed hyperbolic distance-like quantity into a logit.

The visualization script `viz_mlr.py` samples points satisfying an approximate orthogonality constraint:

$$
\langle -p \oplus_c x, a\rangle \approx 0
$$

Those points form the hyperbolic decision surface.

## 13. Loss Function

The base objective is cross entropy:

$$
\mathcal{L}_{CE}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(\ell_{i,y_i})}
{\sum_k \exp(\ell_{i,k})}
$$

If `reg_beta > 0`, the model adds a two-class distance regularizer:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+ \beta
\frac{1}{N}
\sum_i
(y_i - 0.5)d(s_{1,i},s_{2,i})^2
$$

This encourages the sentence distance to encode class structure. For binary labels, examples with different labels can push distances in opposite directions depending on label encoding.

## 14. Optimization

The implementation separates parameters into Euclidean and hyperbolic groups:

- Euclidean parameters use Adam.
- Hyperbolic parameters use manual RSGD or projected SGD.

For Poincare-ball RSGD, the Euclidean gradient is converted to a Riemannian gradient using the inverse metric factor:

$$
\nabla_R \mathcal{L}(x)
=
\frac{1}{(\lambda_x^c)^2}
\nabla_E \mathcal{L}(x)
$$

Since:

$$
\lambda_x^c = \frac{2}{1-c\|x\|^2}
$$

then:

$$
\frac{1}{(\lambda_x^c)^2}
=
\frac{(1-c\|x\|^2)^2}{4}
$$

This is implemented by `riemannian_gradient_c`.

The RSGD update is:

$$
x_{t+1}
=
\exp_{x_t}^c
\left(
-\eta \nabla_R \mathcal{L}(x_t)
\right)
$$

Projected SGD instead takes an ambient step and projects back into the ball:

$$
x_{t+1}
=
\operatorname{proj}
\left(
x_t - \eta \nabla_R \mathcal{L}(x_t)
\right)
$$

The code also clips gradients before the hyperbolic update.

## 15. Why Hyperbolic Geometry Helps

Hyperbolic space grows volume exponentially with radius. This makes it useful for tree-like or hierarchical structure.

For an approximate radial coordinate `r`, available volume grows like:

$$
\operatorname{Vol}(B(r)) \propto \sinh^{n-1}(\sqrt{c}r)
$$

For large `r`:

$$
\sinh(r) \approx \frac{e^r}{2}
$$

so there is exponentially more representational room near the boundary. Sentence meanings or classes with hierarchical relations can therefore separate with lower distortion than in a same-dimensional Euclidean space.

## 16. Reference Concepts

- Poincare ball model: conformal model of hyperbolic space inside an open Euclidean ball.
- Riemannian metric: smoothly varying inner product over tangent spaces.
- Conformal factor: local scale factor relating Euclidean and hyperbolic lengths.
- Geodesic: shortest path under the Riemannian metric.
- Exponential map: maps a tangent vector to the manifold along a geodesic.
- Logarithmic map: inverse of the exponential map locally.
- Gyrovector space: algebraic framework where Mobius addition plays the role of vector addition.
- Riemannian SGD: gradient descent that respects manifold geometry.

## 17. Papers And References

- Ganea, Becigneul, and Hofmann, "Hyperbolic Neural Networks", NeurIPS 2018.
- Nickel and Kiela, "Poincare Embeddings for Learning Hierarchical Representations", NeurIPS 2017.
- Bonnabel, "Stochastic Gradient Descent on Riemannian Manifolds", IEEE Transactions on Automatic Control, 2013.
- Ungar, "Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity", 2008.

