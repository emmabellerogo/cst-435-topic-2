"""Concepts tab: forward propagation, backpropagation, matrix shapes, XOR and
activation functions, explained with the Income Insight network itself.

Everything here runs in the browser session with NumPy only; it never calls
the model. The XOR demo trains a tiny separate network for teaching purposes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui_services as svc

# Used only if /version is unreachable; the live values replace these.
FALLBACK_N_ENCODED = 84
FALLBACK_HIDDEN = [64, 32]
FALLBACK_ACTIVATION = "gelu"

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=float)

_erf = np.vectorize(math.erf)


# ---------------------------------------------------------------------------
# activation functions (NumPy, used for the plots and the XOR demo)
# ---------------------------------------------------------------------------
def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def std_normal_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def std_normal_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def gelu(x):
    """Exact GELU: x * Phi(x), where Phi is the standard normal CDF (what nn.GELU computes)."""
    x = np.asarray(x, dtype=float)
    return x * std_normal_cdf(x)


def gelu_grad(x):
    x = np.asarray(x, dtype=float)
    return std_normal_cdf(x) + x * std_normal_pdf(x)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


ACTS = {"ReLU": (relu, relu_grad), "GELU": (gelu, gelu_grad)}


# ---------------------------------------------------------------------------
# XOR
# ---------------------------------------------------------------------------
def hand_built_xor(X: np.ndarray = XOR_X) -> Dict[str, np.ndarray]:
    """A 2-2-1 ReLU network with hand-picked weights that computes XOR exactly.

    h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 - 1), y = h1 - 2*h2
    """
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([[1.0], [-2.0]])
    b2 = np.array([0.0])
    A1 = X @ W1 + b1
    H1 = relu(A1)
    out = H1 @ W2 + b2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "A1": A1, "H1": H1, "out": out}


def xor_backprop_step(lr: float = 0.5) -> Dict[str, np.ndarray]:
    """One full-batch gradient-descent step on XOR for a fixed 2-2-1 ReLU network.

    A teaching example only (ReLU, plain sigmoid, no dropout, no calibration),
    NOT the served GELU model. The starting weights already classify all four
    points correctly but with low confidence, so the step has non-zero gradients,
    and no pre-activation is exactly 0, so ReLU'(A1) is unambiguous.

    Forward:  A1 = X W1 + b1, H1 = ReLU(A1), z = H1 W2 + b2, q = sigmoid(z)
    Backward: d2 = (q - y)/n, dW2 = H1^T d2, db2 = sum(d2), dH1 = d2 W2^T,
              d1 = dH1 * ReLU'(A1), dW1 = X^T d1, db1 = sum(d1)
    """
    X, y = XOR_X, XOR_Y
    n = len(X)
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([-0.5, -1.5])
    W2 = np.array([[2.0], [-6.0]])
    b2 = np.array([-0.5])

    def forward(W1, b1, W2, b2):
        A1 = X @ W1 + b1
        H1 = relu(A1)
        z = H1 @ W2 + b2
        q = sigmoid(z)
        loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        return A1, H1, z, q, loss

    A1, H1, z, q, loss = forward(W1, b1, W2, b2)
    d2 = (q - y) / n
    dW2 = H1.T @ d2
    db2 = d2.sum(axis=0)
    dH1 = d2 @ W2.T
    d1 = dH1 * relu_grad(A1)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)
    new = {"W1": W1 - lr * dW1, "b1": b1 - lr * db1, "W2": W2 - lr * dW2, "b2": b2 - lr * db2}
    *_, q_after, loss_after = forward(new["W1"], new["b1"], new["W2"], new["b2"])
    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "lr": np.array(lr),
        "A1": A1, "H1": H1, "z": z, "q": q, "loss": np.array(loss),
        "d2": d2, "dW2": dW2, "db2": db2, "dH1": dH1, "d1": d1, "dW1": dW1, "db1": db1,
        "W1_new": new["W1"], "b1_new": new["b1"], "W2_new": new["W2"], "b2_new": new["b2"],
        "q_after": q_after, "loss_after": np.array(loss_after),
    }


def _bmatrix(a, digits: int = 4) -> str:
    """A NumPy array as a LaTeX bmatrix (vectors are shown as one row)."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    def fmt(v):
        v = round(float(v), digits)
        return f"{0.0 if v == 0 else v:g}"
    body = r" \\ ".join(" & ".join(fmt(v) for v in row) for row in a)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def xor_table(hb: Dict[str, np.ndarray]) -> pd.DataFrame:
    """The hand-built network's values, one plain Python int per cell (no NumPy reprs)."""
    def col(a):
        return [int(v) for v in np.asarray(a).tolist()]
    return pd.DataFrame({
        "x1": col(XOR_X[:, 0]),
        "x2": col(XOR_X[:, 1]),
        "a1 = x1 + x2": col(hb["A1"][:, 0]),
        "a2 = x1 + x2 − 1": col(hb["A1"][:, 1]),
        "h1 = ReLU(a1)": col(hb["H1"][:, 0]),
        "h2 = ReLU(a2)": col(hb["H1"][:, 1]),
        "output h1 − 2·h2": col(hb["out"][:, 0]),
        "XOR target": col(XOR_Y[:, 0]),
    })


def train_xor(activation: str = "GELU", hidden: int = 4, lr: float = 0.5,
              epochs: int = 3000, seed: int = 0) -> Tuple[List[float], np.ndarray]:
    """Full-batch gradient descent on XOR with a 2-hidden-1 network.

    Forward:  A1 = X W1 + b1,  H1 = act(A1),  z = H1 W2 + b2,  q = sigmoid(z)
    Loss:     mean binary cross-entropy of q
    Backward: dz = (q - y)/n,  dW2 = H1^T dz,  dA1 = (dz W2^T) * act'(A1),  dW1 = X^T dA1
    (No temperature here: like any training run, it uses the plain sigmoid.)
    Returns (loss per epoch, final probabilities for the 4 inputs).
    """
    act, act_grad = ACTS[activation]
    rng = np.random.default_rng(seed)
    X, y = XOR_X, XOR_Y
    n = len(X)
    W1 = rng.normal(0.0, 1.0, (2, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, 1.0, (hidden, 1))
    b2 = np.zeros(1)
    eps = 1e-12
    losses: List[float] = []
    p = np.zeros_like(y)
    for _ in range(epochs):
        A1 = X @ W1 + b1
        H1 = act(A1)
        z = H1 @ W2 + b2
        p = sigmoid(z)
        losses.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
        dz = (p - y) / n
        dW2 = H1.T @ dz
        db2 = dz.sum(axis=0)
        dA1 = (dz @ W2.T) * act_grad(A1)
        dW1 = X.T @ dA1
        db1 = dA1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    p = sigmoid(act(X @ W1 + b1) @ W2 + b2)
    return losses, p


@st.cache_data(show_spinner=False)
def _cached_train_xor(activation: str, hidden: int, lr: float, epochs: int, seed: int):
    # Streamlit reruns the whole script on every click; cache so the demo is
    # not retrained when another tab is used.
    return train_xor(activation, hidden, lr, epochs, seed)


# ---------------------------------------------------------------------------
# shapes of the served network
# ---------------------------------------------------------------------------
def layer_shapes(n_in: int, hidden: List[int]) -> pd.DataFrame:
    rows = []
    prev = n_in
    widths = list(hidden) + [1]
    for i, h in enumerate(widths, start=1):
        is_out = i == len(widths)
        rows.append({
            "layer": f"output (layer {i})" if is_out else f"hidden {i}",
            "weight W": f"{prev} × {h}",
            "bias b": f"{h}",
            "output for a batch of B rows": f"B × {h}" + (" (one logit per row)" if is_out else ""),
            "parameters": prev * h + h,
        })
        prev = h
    return pd.DataFrame(rows)


def gradient_shapes(n_in: int, hidden: List[int]) -> pd.DataFrame:
    """Shape of every backpropagation quantity, output layer first (the order it is computed)."""
    widths = list(hidden) + [1]
    ins = [n_in] + list(hidden)
    names = ["X"] + [f"H{i}" for i in range(1, len(hidden) + 1)]
    rows = []
    for i in range(len(widths), 0, -1):
        h, prev, src = widths[i - 1], ins[i - 1], names[i - 1]
        is_out = i == len(widths)
        rows.append({
            "layer": f"output (layer {i})" if is_out else f"hidden {i}",
            "error signal δ": f"δ{i} = " + ("∂L/∂z" if is_out else f"∂L/∂A{i}") + f": B × {h}",
            "weight gradient": f"∂L/∂W{i} = {src}ᵀ δ{i}: ({prev} × B)(B × {h}) = {prev} × {h}",
            "bias gradient": f"∂L/∂b{i} = column sums of δ{i}: {h}",
        })
    return pd.DataFrame(rows)


def _model_shape(url: Optional[str]) -> Tuple[int, List[int], str, Optional[float]]:
    version = svc.try_version(url)
    model = (version or {}).get("model") or {}
    return (
        int(model.get("n_encoded_features") or FALLBACK_N_ENCODED),
        list(model.get("hidden_sizes") or FALLBACK_HIDDEN),
        str(model.get("activation") or FALLBACK_ACTIVATION),
        model.get("temperature"),
    )


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def render_xor_backprop_step() -> None:
    s = xor_backprop_step()
    m = _bmatrix
    st.markdown("**One backpropagation step by hand (numerical XOR example)**")
    st.caption(
        "A separate 2-2-1 teaching network: ReLU hidden layer, plain sigmoid q = σ(z), no "
        "dropout and no temperature. It is not the served GELU model. The starting weights "
        "already put every point on the right side of 0.5, but only weakly, so the gradients "
        "are not zero. All four XOR rows form one batch (B = 4); values are rounded to 4 decimals."
    )
    st.latex(r"W_1 = " + m(s["W1"]) + r",\; b_1 = " + m(s["b1"]) + r",\; W_2 = " + m(s["W2"])
             + r",\; b_2 = " + m(s["b2"]) + r",\; \eta = " + f"{float(s['lr']):g}")
    st.markdown("*Forward pass* (X is 4 × 2, so A₁ and H₁ are 4 × 2 and z, q are 4 × 1):")
    st.latex(r"A_1 = XW_1 + b_1 = " + m(s["A1"]) + r",\quad H_1 = \mathrm{ReLU}(A_1) = "
             + m(s["H1"]))
    st.latex(r"z = H_1W_2 + b_2 = " + m(s["z"]) + r",\quad q = \sigma(z) = " + m(s["q"])
             + r",\quad y = " + m(XOR_Y))
    st.latex(r"\mathcal{L} = -\tfrac{1}{4}\textstyle\sum_i\big[y_i\log q_i + (1-y_i)\log(1-q_i)\big]"
             r" = " + f"{float(s['loss']):.4f}")
    st.markdown("*Backward pass* (each gradient has the shape of what it differentiates):")
    st.latex(r"\delta_2 = \tfrac{1}{4}(q - y) = " + m(s["d2"]) + r",\quad "
             r"\tfrac{\partial\mathcal{L}}{\partial W_2} = H_1^{\top}\delta_2 = " + m(s["dW2"])
             + r",\quad \tfrac{\partial\mathcal{L}}{\partial b_2} = " + m(s["db2"]))
    st.latex(r"\tfrac{\partial\mathcal{L}}{\partial H_1} = \delta_2W_2^{\top} = " + m(s["dH1"])
             + r",\quad \mathrm{ReLU}'(A_1) = " + m(relu_grad(s["A1"])))
    st.latex(r"\delta_1 = \tfrac{\partial\mathcal{L}}{\partial H_1}\odot\mathrm{ReLU}'(A_1) = "
             + m(s["d1"]))
    st.latex(r"\tfrac{\partial\mathcal{L}}{\partial W_1} = X^{\top}\delta_1 = " + m(s["dW1"])
             + r",\quad \tfrac{\partial\mathcal{L}}{\partial b_1} = \mathbf{1}^{\top}\delta_1 = "
             + m(s["db1"]))
    st.markdown("*Update* (plain gradient descent, W ← W − η ∂L/∂W):")
    st.latex(r"W_2 \leftarrow " + m(s["W2"]) + r" - 0.5" + m(s["dW2"]) + r" = " + m(s["W2_new"])
             + r",\quad b_1 \leftarrow " + m(s["b1_new"]) + r",\quad W_1 \leftarrow "
             + m(s["W1_new"]))
    st.write(
        f"After this one step the loss falls from {float(s['loss']):.4f} to "
        f"{float(s['loss_after']):.4f}. Notice the effect of ReLU: the first row (0, 0) has "
        "A₁ < 0 in both units, so ReLU′ = 0 there and that row sends no gradient to W₁. The "
        "first hidden unit's incoming weights get a zero gradient overall, because its "
        "contributions from the other three rows cancel. The served model is trained the same "
        "way, but with GELU, dropout, mini-batches and AdamW instead of this plain step."
    )


def render(url: Optional[str]) -> None:
    n_in, hidden, act_name, temperature = _model_shape(url)
    h1, h2 = (hidden + [None, None])[:2]

    st.header("How the network works")
    st.write(
        "Income Insight uses a **multi-layer perceptron (MLP)**: a stack of fully connected "
        "layers. Each person's 10 raw features are turned into numbers by the preprocessing "
        f"step (scaling and one-hot encoding), giving a vector of **{n_in}** values. The network "
        "maps that vector to one number, the probability that income is above \\$50K."
    )

    # -- forward propagation ------------------------------------------------
    st.subheader("1. Forward propagation")
    st.write(
        "Forward propagation means pushing the input through the layers, one matrix "
        "multiplication at a time. We process a **batch** of B people at once, so the input "
        f"is a matrix **X** with B rows and {n_in} columns."
    )
    st.latex(r"A_1 = X W_1 + b_1, \qquad H_1 = \phi(A_1)")
    st.latex(r"A_2 = H_1 W_2 + b_2, \qquad H_2 = \phi(A_2)")
    st.latex(r"z = H_2 W_3 + b_3 \qquad (B \times 1\text{: one logit per row})")
    st.write(
        f"Here φ is the activation function (this model uses **{act_name.upper()}**) and σ is "
        "the sigmoid σ(u) = 1 / (1 + e^(−u)), which squeezes any number into (0, 1). The bias "
        "vector b is added to every row (broadcasting). The logit z is turned into a "
        "probability in two different ways, one while training and one when serving:"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Training** (fits the weights)")
        st.latex(r"q = \sigma(z)")
        st.caption("The plain sigmoid. The loss and every gradient below are written in terms "
                   "of q. During training only, **dropout** randomly zeroes 10% of the hidden "
                   "values after each activation.")
    with right:
        st.markdown("**Inference** (what the API returns)")
        st.latex(r"p = \sigma\!\left(\frac{z}{T}\right)")
        st.caption("The calibrated probability. T is fitted **after** training, on the "
                   "validation set, with the weights frozen"
                   + (f" (T ≈ {temperature:.4f})" if temperature else "")
                   + ". Dropout is off.")

    st.markdown("**Matrix shapes for the model being served**")
    shapes = layer_shapes(n_in, hidden)
    st.dataframe(shapes, hide_index=True, use_container_width=True)
    st.caption(
        f"Total trainable parameters: {int(shapes['parameters'].sum()):,}. "
        "The inner dimensions always match: (B × "
        f"{n_in}) · ({n_in} × {h1}) gives (B × {h1}), and so on."
    )

    # -- backpropagation ------------------------------------------------------
    st.subheader("2. Backpropagation (training only)")
    st.write(
        "Training adjusts the weights to make the **binary cross-entropy (BCE)** loss small. "
        "For labels y ∈ {0, 1} (1 means >50K) and the training probability q = σ(z):"
    )
    st.latex(r"\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\Big[y_i\log q_i + (1-y_i)\log(1-q_i)\Big]")
    st.write(
        "Backpropagation applies the chain rule from the output back to the input, reusing "
        "each layer's result for the layer before it. Because σ′(z) = q(1 − q), the sigmoid's "
        "derivative cancels against the BCE's, and the error signal at the output is simply "
        "q − y, averaged over the batch. Then, layer by layer (⊙ means element-wise "
        "multiplication):"
    )
    st.latex(r"\delta_3 = \frac{\partial \mathcal{L}}{\partial z} = \frac{1}{B}(q - y)"
             r"\quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial W_3} = H_2^{\top}\delta_3")
    st.latex(r"\delta_2 = \frac{\partial \mathcal{L}}{\partial A_2} = (\delta_3 W_3^{\top}) \odot \phi'(A_2)"
             r"\quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial W_2} = H_1^{\top}\delta_2")
    st.latex(r"\delta_1 = \frac{\partial \mathcal{L}}{\partial A_1} = (\delta_2 W_2^{\top}) \odot \phi'(A_1)"
             r"\quad\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial W_1} = X^{\top}\delta_1")
    with st.expander("The chain-rule steps behind these three lines", expanded=True):
        st.write(
            f"Shapes for the served model, with a batch of B rows: X is B × {n_in}; "
            f"A₁ and H₁ are B × {h1}; A₂ and H₂ are B × {h2}; z, q, y and δ₃ are B × 1. "
            f"W₁ is {n_in} × {h1}, W₂ is {h1} × {h2}, W₃ is {h2} × 1, and the biases have "
            f"{h1}, {h2} and 1 entries. Each step below multiplies one local derivative into "
            "the error coming from the layer above."
        )
        st.markdown("**Step 1: loss → logit** (per row i; the 1/B comes from the mean)")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial q_i} = -\frac{1}{B}\left(\frac{y_i}{q_i}"
                 r" - \frac{1-y_i}{1-q_i}\right) = \frac{1}{B}\,\frac{q_i - y_i}{q_i(1-q_i)},"
                 r"\qquad \frac{\partial q_i}{\partial z_i} = \sigma'(z_i) = q_i(1-q_i)")
        st.latex(r"\Rightarrow\; \delta_{3,i} = \frac{\partial \mathcal{L}}{\partial q_i}\cdot"
                 r"\frac{\partial q_i}{\partial z_i} = \frac{q_i - y_i}{B}")
        st.markdown("**Step 2: logit → output weights and bias.** Since "
                    "z = H₂W₃ + 1b₃ (1 is a column of B ones), each entry is "
                    "zᵢ = Σⱼ H₂[i, j] W₃[j] + b₃, so")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial W_3[j]} = \sum_{i=1}^{B} H_2[i,j]\,\delta_{3,i}"
                 r"\;\Rightarrow\; \frac{\partial \mathcal{L}}{\partial W_3} = "
                 r"\underbrace{H_2^{\top}}_{" + f"{h2}" + r"\times B}\underbrace{\delta_3}_{B\times 1},"
                 r"\qquad \frac{\partial \mathcal{L}}{\partial b_3} = \mathbf{1}^{\top}\delta_3")
        st.markdown("**Step 3: logit → previous layer's output, then through the activation.** "
                    "z depends on H₂[i, j] through W₃[j], and H₂ = φ(A₂) is applied entry by "
                    "entry, so its Jacobian is diagonal and becomes an element-wise product:")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial H_2} = \underbrace{\delta_3}_{B\times 1}"
                 r"\underbrace{W_3^{\top}}_{1\times " + f"{h2}" + r"},\qquad "
                 r"\delta_2 = \frac{\partial \mathcal{L}}{\partial H_2}\odot\phi'(A_2)\quad(B\times "
                 + f"{h2}" + r")")
        st.markdown("**Step 4: repeat.** Steps 2 and 3 are the same for every layer, which is "
                    "why backpropagation is a loop:")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial W_2} = H_1^{\top}\delta_2\;(" + f"{h1}"
                 + r"\times" + f"{h2}" + r"),\quad \delta_1 = (\delta_2W_2^{\top})\odot\phi'(A_1)"
                 r"\;(B\times" + f"{h1}" + r"),\quad \frac{\partial \mathcal{L}}{\partial W_1} = "
                 r"X^{\top}\delta_1\;(" + f"{n_in}" + r"\times" + f"{h1}" + r")")
        st.caption(f"For this model φ is {act_name.upper()}"
                   + (", so φ′(a) = Φ(a) + a·N(a), where Φ and N are the standard normal CDF and "
                      "density)." if act_name.lower() == "gelu" else "."))

    st.markdown("**Gradient shapes for the model being served** (computed output first)")
    st.dataframe(gradient_shapes(n_in, hidden), hide_index=True, use_container_width=True)
    st.write(
        "Each bias gradient is the column sum of its δ. Every gradient has exactly the same "
        f"shape as the weight it updates: for example Xᵀ is ({n_in} × B) and δ₁ is (B × {h1}), "
        f"so ∂L/∂W₁ is ({n_in} × {h1}), the same as W₁. With dropout on, H₁ and H₂ are the "
        "masked values and the same mask multiplies the δ flowing back through them. The "
        "optimizer (AdamW) then moves each weight a small step against its gradient. One pass "
        "over the training data is one **epoch**; the saved model is the epoch with the lowest "
        "validation loss. (In PyTorch, `BCEWithLogitsLoss` takes z directly and applies "
        "q = σ(z) inside the loss, which is numerically safer but gives the same gradient.)"
    )
    st.info(
        "Calibration happens afterwards and is separate. Once training has finished, the "
        "weights are frozen and only T is fitted, by minimizing BCE of p = σ(z/T) on the "
        "**validation** set. T never enters the backpropagation above. Dividing by T > 0 never "
        "changes which side of 0.5 a prediction falls on, so it changes confidence, not labels."
    )

    # -- XOR ------------------------------------------------------------------
    st.subheader("3. Why hidden layers matter: XOR")
    st.write(
        "XOR (exclusive or) outputs 1 when exactly one of its two inputs is 1. No single "
        "straight line can separate the 1s from the 0s, so a model with no hidden layer "
        "(logistic regression) cannot learn it. One hidden layer with a non-linear activation "
        "can: it bends the input space so that a straight line works afterwards."
    )
    st.markdown(
        "**Why one line cannot work.** A single linear boundary predicts 1 when "
        "w₁x₁ + w₂x₂ + b > 0. XOR would need b < 0 for (0, 0), w₁ + b > 0 for (1, 0), "
        "w₂ + b > 0 for (0, 1) and w₁ + w₂ + b < 0 for (1, 1). Adding the two middle "
        "inequalities gives w₁ + w₂ + 2b > 0, so w₁ + w₂ + b > −b > 0, which contradicts the "
        "last one. No choice of weights works.\n\n"
        "**Why stacking matrices alone does not help.** Without an activation, two layers "
        "collapse into one: (XW₁ + b₁)W₂ + b₂ = X(W₁W₂) + (b₁W₂ + b₂), which is still one "
        "linear boundary. Each matrix multiplication can only rotate, stretch and project the "
        "inputs. The non-linear φ between them is what lets the next matrix multiplication "
        "draw a boundary that is curved in the original inputs. More hidden units and layers "
        "give more such pieces, and that is where the network's expressive capacity comes from."
    )
    hb = hand_built_xor()
    st.markdown("**A hand-built network that solves XOR**")
    st.latex(r"h_1 = \mathrm{ReLU}(x_1 + x_2), \quad h_2 = \mathrm{ReLU}(x_1 + x_2 - 1), "
             r"\quad \hat y = h_1 - 2h_2")
    st.latex(r"W_1 = \begin{bmatrix}1 & 1\\ 1 & 1\end{bmatrix},\; b_1 = \begin{bmatrix}0 & -1\end{bmatrix},"
             r"\; W_2 = \begin{bmatrix}1\\ -2\end{bmatrix},\; b_2 = 0")
    st.dataframe(xor_table(hb), hide_index=True, use_container_width=True)
    st.caption(
        "Shapes: X is 4 × 2, W1 is 2 × 2, so H1 is 4 × 2; W2 is 2 × 1, so the output is 4 × 1. "
        "The second hidden unit only switches on for (1, 1), and subtracting it twice "
        "cancels the first unit there."
    )

    render_xor_backprop_step()

    st.markdown("**Now let backpropagation find the weights itself**")
    c1, c2, c3, c4 = st.columns(4)
    act = c1.selectbox("Activation", list(ACTS), index=1, key="xor_act")
    hidden_units = c2.slider("Hidden units", 2, 8, 4, key="xor_hidden")
    lr = c3.select_slider("Learning rate", options=[0.05, 0.1, 0.5, 1.0], value=0.5, key="xor_lr")
    seed = c4.number_input("Random seed", min_value=0, max_value=999, value=0, step=1, key="xor_seed")
    losses, probs = _cached_train_xor(act, int(hidden_units), float(lr), 3000, int(seed))
    curve = pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "BCE loss": losses})
    st.altair_chart(
        alt.Chart(curve).mark_line().encode(
            x=alt.X("epoch:Q", title="epoch"),
            y=alt.Y("BCE loss:Q", title="training loss (BCE)"),
        ).properties(height=220),
        use_container_width=True,
    )
    result = pd.DataFrame({
        "x1": [int(v) for v in XOR_X[:, 0].tolist()],
        "x2": [int(v) for v in XOR_X[:, 1].tolist()],
        "target": [int(v) for v in XOR_Y[:, 0].tolist()],
        "predicted probability q": [round(float(v), 3) for v in probs[:, 0].tolist()],
        "predicted label (q ≥ 0.5)": [int(v >= 0.5) for v in probs[:, 0].tolist()],
    })
    st.dataframe(result, hide_index=True, use_container_width=True)
    solved = bool(((probs[:, 0] >= 0.5).astype(int) == XOR_Y[:, 0].astype(int)).all())
    if solved:
        st.success(f"Solved: all four XOR cases are correct. Final loss {losses[-1]:.4f}.")
    else:
        st.warning(
            f"Not solved with this seed (final loss {losses[-1]:.4f}). With ReLU, a hidden unit "
            "whose input is negative for all four points outputs 0 and gets zero gradient, so it "
            "stops learning (a 'dead' unit). Try another seed, more hidden units or GELU."
        )

    # -- activations ----------------------------------------------------------
    st.subheader("4. Activation functions: ReLU vs GELU")
    st.markdown(
        "- **ReLU = Rectified Linear Unit**: ReLU(x) = max(0, x). It sharply sets every "
        "negative value to zero and passes positive values through unchanged. It is cheap and "
        "works well, but its gradient is exactly 0 for negative inputs, so a unit can stop "
        "learning.\n"
        "- **GELU = Gaussian Error Linear Unit**: GELU(x) = x · Φ(x), where Φ is the standard "
        "normal cumulative distribution function. Instead of a hard cut at 0, it scales each "
        "input by the probability that a standard normal value is below it. It is **smooth**, "
        "lets small negative values through (its minimum is about −0.17, near x ≈ −0.75), and "
        "keeps a non-zero gradient for moderately negative inputs."
    )
    show_grad = st.toggle("Show derivatives instead (what backpropagation multiplies by)",
                          key="act_grad")
    xs = np.linspace(-4, 4, 161)
    if show_grad:
        df = pd.DataFrame({"x": xs, "ReLU'": relu_grad(xs), "GELU'": gelu_grad(xs)})
    else:
        df = pd.DataFrame({"x": xs, "ReLU": relu(xs), "GELU": gelu(xs)})
    long = df.melt("x", var_name="function", value_name="value")
    st.altair_chart(
        alt.Chart(long).mark_line().encode(
            x="x:Q", y=alt.Y("value:Q", title="derivative" if show_grad else "output"),
            color=alt.Color("function:N", title=None),
        ).properties(height=260),
        use_container_width=True,
    )

    perf = svc.try_performance()
    if perf:
        by_name = {e["name"]: e for e in perf["experiments"]}
        base, gel = by_name.get("baseline"), by_name.get("gelu")
        if base and gel and base.get("val_loss") is not None and gel.get("val_loss") is not None:
            st.write(
                "**In our controlled experiment**, the only difference between the `baseline` and "
                "`gelu` configurations was the activation (both [64, 32], dropout 0.1, same seed, "
                f"split and budget). Validation loss was {base['val_loss']:.4f} with ReLU and "
                f"{gel['val_loss']:.4f} with GELU. GELU was slightly better, but the gap is small "
                "and comes from a single seed, so it is weak evidence that GELU is better in general."
            )
