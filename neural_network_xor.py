"""
╔══════════════════════════════════════════════════════════════╗
║         NEURAL NETWORK FROM SCRATCH — XOR PROBLEM           ║
║              Pure NumPy · No ML libraries                    ║
╚══════════════════════════════════════════════════════════════╝

The XOR problem is the "Hello World" of neural networks.
A single neuron (linear model) CANNOT learn XOR — it's not
linearly separable. A hidden layer with non-linear activations
solves it. This makes XOR the perfect minimal proof of concept.

Truth table:
  Input A | Input B | Output (A XOR B)
  --------|---------|----------------
     0    |    0    |       0
     0    |    1    |       1
     1    |    0    |       1
     1    |    1    |       0

Network Architecture:
  Input Layer  → 2 neurons  (one per input bit)
  Hidden Layer → 4 neurons  (with sigmoid activation)
  Output Layer → 1 neuron   (with sigmoid activation → 0 or 1)
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# Seed for reproducibility
# ──────────────────────────────────────────────────────────────
np.random.seed(42)


# ══════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def sigmoid(z):
    """
    Sigmoid (logistic) activation: maps any real number → (0, 1).

    Formula:  σ(z) = 1 / (1 + e^(-z))

    Why use it?
      - Outputs are interpretable as probabilities.
      - Smooth + differentiable (needed for backprop).
      - Non-linear → lets the network learn XOR.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid WITH RESPECT TO its input z.

    Formula:  σ'(z) = σ(z) · (1 - σ(z))

    Used in backpropagation to compute how much each
    neuron's pre-activation value contributed to the error.
    """
    s = sigmoid(z)
    return s * (1 - s)


# ══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss — measures how wrong the predictions are.

    Formula:  L = -[ y·log(ŷ) + (1-y)·log(1-ŷ) ]

    Why this instead of MSE?
      - BCE penalises confident wrong predictions much harder.
      - Mathematically paired with sigmoid (clean gradient).
      - Standard choice for binary classification tasks.

    Clipping prevents log(0) = -infinity.
    """
    eps = 1e-15  # tiny value to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ══════════════════════════════════════════════════════════════
# NEURAL NETWORK CLASS
# ══════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    A 3-layer fully connected neural network:

      [Input: 2] → [Hidden: 4, sigmoid] → [Output: 1, sigmoid]

    Parameters
    ----------
    input_size   : number of input features (2 for XOR)
    hidden_size  : number of hidden neurons  (4 gives plenty of capacity)
    output_size  : number of output neurons  (1 for binary classification)
    learning_rate: how big each gradient descent step is (η)
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
        self.lr = learning_rate

        # ── Weight Initialisation ──────────────────────────────
        # Small random weights break symmetry — if all weights were equal,
        # every neuron would learn the same thing (useless!).
        #
        # Shape of W1: (input_size  × hidden_size) = (2 × 4)
        # Shape of W2: (hidden_size × output_size) = (4 × 1)
        #
        # Xavier / Glorot scaling (÷ √fan_in) keeps activations from
        # exploding or vanishing at the start of training.

        self.W1 = np.random.randn(input_size,  hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))    # bias for hidden layer

        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))    # bias for output layer

        # Storage for training history
        self.loss_history = []

    # ──────────────────────────────────────────────────────────
    # FORWARD PASS
    # ──────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Compute a prediction for input X.

        Step 1 — Hidden layer pre-activation (linear combination):
            Z1 = X · W1 + b1

        Step 2 — Hidden layer activation (introduce non-linearity):
            A1 = sigmoid(Z1)

        Step 3 — Output layer pre-activation:
            Z2 = A1 · W2 + b2

        Step 4 — Output activation (squash to 0–1):
            A2 = sigmoid(Z2)   ← this is our prediction ŷ

        We store Z1, A1, Z2 because backprop needs them.
        """
        self.Z1 = X @ self.W1 + self.b1      # (N × hidden_size)
        self.A1 = sigmoid(self.Z1)            # (N × hidden_size)

        self.Z2 = self.A1 @ self.W2 + self.b2  # (N × output_size)
        self.A2 = sigmoid(self.Z2)              # (N × output_size)  ← ŷ

        return self.A2

    # ──────────────────────────────────────────────────────────
    # BACKWARD PASS (Backpropagation)
    # ──────────────────────────────────────────────────────────

    def backward(self, X, y):
        """
        Compute gradients using the chain rule, then update weights.

        The chain rule lets us figure out how much each weight
        contributed to the final loss — working backwards from
        the output layer to the input layer.

        ── Output layer gradients ────────────────────────────
        dL/dZ2 = A2 - y
          (this elegant result comes from combining BCE loss
           derivative with sigmoid derivative — they cancel nicely)

        dL/dW2 = A1ᵀ · dZ2         (chain rule through Z2 = A1·W2)
        dL/db2 = sum(dZ2)           (bias gradient = sum of errors)

        ── Hidden layer gradients ────────────────────────────
        dL/dA1 = dZ2 · W2ᵀ         (error flowing back through W2)
        dL/dZ1 = dA1 · σ'(Z1)      (chain rule through sigmoid)
        dL/dW1 = Xᵀ · dZ1
        dL/db1 = sum(dZ1)

        ── Gradient Descent Update ───────────────────────────
        W = W - η · dL/dW           (move against the gradient)
        """
        N = X.shape[0]  # number of training samples (for averaging)

        # Output layer
        dZ2 = self.A2 - y                           # (N × 1)
        dW2 = (self.A1.T @ dZ2) / N                 # (hidden × 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N

        # Hidden layer
        dA1 = dZ2 @ self.W2.T                       # (N × hidden)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)     # (N × hidden)
        dW1 = (X.T @ dZ1) / N                       # (input × hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N

        # Gradient descent — update all parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ──────────────────────────────────────────────────────────
    # TRAIN
    # ──────────────────────────────────────────────────────────

    def train(self, X, y, epochs=10_000, print_every=1000):
        """
        Full training loop: forward → loss → backward → update, repeated.

        One full pass over the dataset = one epoch.
        For XOR (4 samples) we train for many epochs because
        each pass is cheap and the problem needs many iterations.
        """
        print("=" * 55)
        print("  Training Neural Network on XOR")
        print("=" * 55)
        print(f"  Architecture : {X.shape[1]} → 4 → 1")
        print(f"  Learning rate: {self.lr}")
        print(f"  Epochs       : {epochs:,}")
        print("=" * 55)

        for epoch in range(1, epochs + 1):
            # Forward pass → get predictions
            y_pred = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Backward pass → update weights
            self.backward(X, y)

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch:>6,} | Loss: {loss:.6f} | Accuracy: {acc*100:.1f}%")

        print("=" * 55)
        print("  Training complete!")
        print("=" * 55)

    # ──────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Return binary predictions (0 or 1) by thresholding at 0.5.
        The network outputs a probability — if >= 0.5 we call it 1.
        """
        return (self.forward(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return raw probability outputs (before thresholding)."""
        return self.forward(X)

    def accuracy(self, X, y):
        """Fraction of predictions that match the true labels."""
        return np.mean(self.predict(X) == y)


# ══════════════════════════════════════════════════════════════
# MAIN — Train and Evaluate
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Dataset: all 4 possible XOR inputs ────────────────────
    X = np.array([
        [0, 0],   # 0 XOR 0 = 0
        [0, 1],   # 0 XOR 1 = 1
        [1, 0],   # 1 XOR 0 = 1
        [1, 1],   # 1 XOR 1 = 0
    ], dtype=float)

    y = np.array([[0], [1], [1], [0]], dtype=float)

    # ── Create and train the network ───────────────────────────
    nn = NeuralNetwork(
        input_size    = 2,
        hidden_size   = 4,
        output_size   = 1,
        learning_rate = 0.5
    )

    nn.train(X, y, epochs=10_000, print_every=1000)

    # ── Final predictions ──────────────────────────────────────
    print("\n  📊 Final Predictions")
    print("  " + "-" * 45)
    print(f"  {'Input A':>8} {'Input B':>8} {'True':>8} {'Pred':>8} {'Prob':>10}")
    print("  " + "-" * 45)

    probs = nn.predict_proba(X)
    preds = nn.predict(X)

    for i in range(len(X)):
        a, b   = int(X[i][0]), int(X[i][1])
        true   = int(y[i][0])
        pred   = int(preds[i][0])
        prob   = float(probs[i][0])
        status = "✅" if pred == true else "❌"
        print(f"  {a:>8} {b:>8} {true:>8} {pred:>8}   {prob:.4f}  {status}")

    print("  " + "-" * 45)

    final_acc = nn.accuracy(X, y)
    print(f"\n  Final Accuracy : {final_acc * 100:.1f}%")
    print(f"  Final Loss     : {nn.loss_history[-1]:.6f}")

    # ── Inspect learned weights ────────────────────────────────
    print("\n  🔬 Learned Weights")
    print("  " + "-" * 45)
    print(f"  W1 (input→hidden):\n{nn.W1.round(3)}")
    print(f"\n  b1 (hidden biases):\n{nn.b1.round(3)}")
    print(f"\n  W2 (hidden→output):\n{nn.W2.round(3)}")
    print(f"\n  b2 (output bias):\n{nn.b2.round(3)}")

    # ── Loss curve (ASCII chart) ───────────────────────────────
    print("\n  📉 Loss Curve (every 1000 epochs)")
    print("  " + "-" * 45)
    samples = nn.loss_history[::1000]
    max_loss = max(samples)
    bar_width = 35
    for i, loss in enumerate(samples):
        epoch  = (i + 1) * 1000 if i > 0 else 1
        filled = int((loss / max_loss) * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        print(f"  Ep {epoch:>5,}  {bar}  {loss:.4f}")

    print("\n  Done. The network has learned XOR! 🎉\n")
