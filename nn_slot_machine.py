"""
Neural Network for Slot Machine Output Prediction
==================================================
Built from scratch using NumPy only.

Architecture:
  Input  : last WINDOW spins, one-hot encoded  (WINDOW * NUM_SYMBOLS * NUM_REELS)
  Hidden : two fully-connected layers with ReLU
  Output : probability over NUM_SYMBOLS for each of the 3 reels (softmax)

Note: slot machines use a PRNG — true prediction is impossible.
      This network learns the *empirical symbol distribution* from history
      and reports confidence alongside each prediction.
"""

import numpy as np
import random
import time
import os

# ── colour helpers ────────────────────────────────────────────────────────────
R="\033[91m"; Y="\033[93m"; G="\033[92m"; C="\033[96m"
M="\033[95m"; B="\033[94m"; W="\033[97m"; D="\033[2m"; BD="\033[1m"; X="\033[0m"

def c(text, col): return f"{col}{text}{X}"
def bar(v, w=20, col=G): return c("█"*int(v*w), col)+c("░"*(w-int(v*w)), D)

# ── slot machine config ───────────────────────────────────────────────────────
SYMBOLS  = ["7️ ", "💎", "🍒", "🍋", "🍊", "🍇", "⭐"]
WEIGHTS  = [2, 3, 6, 8, 9, 10, 12]          # rarer = lower weight
SYM_COLS = [R, C, R, Y, Y, M, Y]
NUM_SYM  = len(SYMBOLS)
NUM_REEL = 3
WINDOW   = 5                                 # how many past spins to use as input

# ── data generation ───────────────────────────────────────────────────────────
def spin():
    return [random.choices(range(NUM_SYM), weights=WEIGHTS, k=1)[0]
            for _ in range(NUM_REEL)]

def generate_data(n=5000):
    spins = [spin() for _ in range(n + WINDOW)]
    X, Y = [], []
    for i in range(WINDOW, n + WINDOW):
        window = spins[i-WINDOW:i]           # shape (WINDOW, NUM_REEL)
        x = np.zeros(WINDOW * NUM_REEL * NUM_SYM)
        for t, s in enumerate(window):
            for r, sym in enumerate(s):
                x[t*NUM_REEL*NUM_SYM + r*NUM_SYM + sym] = 1.0
        X.append(x)
        y = np.array(spins[i])              # (NUM_REEL,)
        Y.append(y)
    return np.array(X), np.array(Y)

# ── neural network ────────────────────────────────────────────────────────────
class NeuralNetwork:
    """
    Two-hidden-layer MLP.
    Output: 3 independent softmax heads (one per reel).
    Loss  : mean cross-entropy across all 3 reels.
    """

    def __init__(self, input_dim, hidden1=256, hidden2=128, lr=0.001):
        self.lr = lr
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden1)
        scale3 = np.sqrt(2.0 / hidden2)

        self.W1 = np.random.randn(input_dim, hidden1) * scale1
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * scale2
        self.b2 = np.zeros(hidden2)
        # 3 output heads
        self.W3 = [np.random.randn(hidden2, NUM_SYM) * scale3 for _ in range(NUM_REEL)]
        self.b3 = [np.zeros(NUM_SYM) for _ in range(NUM_REEL)]

    # activations
    @staticmethod
    def relu(x):      return np.maximum(0, x)
    @staticmethod
    def relu_d(x):    return (x > 0).astype(float)
    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        self.X   = X
        self.z1  = X @ self.W1 + self.b1
        self.a1  = self.relu(self.z1)
        self.z2  = self.a1 @ self.W2 + self.b2
        self.a2  = self.relu(self.z2)
        self.out = [self.softmax(self.a2 @ self.W3[r] + self.b3[r])
                    for r in range(NUM_REEL)]
        return self.out

    def loss(self, Y):
        """Mean cross-entropy."""
        n   = len(Y)
        eps = 1e-9
        total = 0.0
        for r in range(NUM_REEL):
            p      = self.out[r][np.arange(n), Y[:, r]]
            total += -np.log(p + eps).mean()
        return total / NUM_REEL

    def backward(self, Y):
        n  = len(Y)
        da2 = np.zeros_like(self.a2)

        for r in range(NUM_REEL):
            # softmax + cross-entropy gradient
            dz3       = self.out[r].copy()
            dz3[np.arange(n), Y[:, r]] -= 1
            dz3      /= n

            self.W3[r] -= self.lr * self.a2.T @ dz3
            self.b3[r] -= self.lr * dz3.sum(axis=0)
            da2        += dz3 @ self.W3[r].T

        dz2  = da2 * self.relu_d(self.z2)
        self.W2 -= self.lr * self.a1.T @ dz2
        self.b2 -= self.lr * dz2.sum(axis=0)

        da1  = dz2 @ self.W2.T
        dz1  = da1 * self.relu_d(self.z1)
        self.W1 -= self.lr * self.X.T @ dz1
        self.b1 -= self.lr * dz1.sum(axis=0)

    def accuracy(self, Y):
        preds = np.stack([o.argmax(axis=1) for o in self.out], axis=1)
        return (preds == Y).mean()

    def predict_single(self, x):
        """Returns list of (predicted_symbol, confidence) per reel."""
        out = self.forward(x[np.newaxis])
        return [(o[0].argmax(), o[0].max()) for o in out]

# ── training ──────────────────────────────────────────────────────────────────
def train(nn, X_train, Y_train, X_val, Y_val,
          epochs=40, batch=128, patience=8):
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best_val, best_weights, wait = np.inf, None, 0

    for ep in range(1, epochs + 1):
        # shuffle
        idx = np.random.permutation(len(X_train))
        X_s, Y_s = X_train[idx], Y_train[idx]

        for i in range(0, len(X_s), batch):
            nn.forward(X_s[i:i+batch])
            nn.backward(Y_s[i:i+batch])

        # metrics
        nn.forward(X_train)
        tl  = nn.loss(Y_train)
        ta  = nn.accuracy(Y_train)

        nn.forward(X_val)
        vl  = nn.loss(Y_val)
        va  = nn.accuracy(Y_val)

        history["loss"].append(tl)
        history["val_loss"].append(vl)
        history["acc"].append(ta)
        history["val_acc"].append(va)

        # progress bar
        prog  = int((ep / epochs) * 30)
        pbar  = c("█"*prog, G) + c("░"*(30-prog), D)
        print(f"\r  Epoch {ep:>3}/{epochs}  [{pbar}]  "
              f"loss={c(f'{tl:.4f}',Y)}  val_loss={c(f'{vl:.4f}',C)}  "
              f"acc={c(f'{ta*100:.1f}%',G)}  val_acc={c(f'{va*100:.1f}%',M)}",
              end="", flush=True)

        # early stopping
        if vl < best_val:
            best_val = vl
            best_weights = (
                nn.W1.copy(), nn.b1.copy(),
                nn.W2.copy(), nn.b2.copy(),
                [w.copy() for w in nn.W3],
                [b.copy() for b in nn.b3]
            )
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  {c('Early stop at epoch '+str(ep), Y)}")
                break

    # restore best
    nn.W1, nn.b1, nn.W2, nn.b2, nn.W3, nn.b3 = best_weights
    print()
    return history

# ── pretty print ──────────────────────────────────────────────────────────────
def print_header():
    print(c("╔══════════════════════════════════════════════════╗", Y))
    print(c("║", Y) + c("   🧠  SLOT MACHINE NEURAL NETWORK PREDICTOR   ", BD+W) + c("║", Y))
    print(c("╚══════════════════════════════════════════════════╝\n", Y))

def show_history(h):
    print(c("\n  ── Training Curve ────────────────────────────────", D+W))
    epochs = len(h["loss"])
    step   = max(1, epochs // 10)
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
    print(c("  " + "─"*42, D))
    for i in range(0, epochs, step):
        vl  = h["val_loss"][i]
        col = G if vl < 1.8 else Y if vl < 1.9 else R
        print(f"  {i+1:>5}  {h['loss'][i]:>10.4f}  "
              f"{c(f'{vl:.4f}', col):>20}  "
              f"{h['val_acc'][i]*100:>7.1f}%")

def show_distribution(nn, X_val, Y_val):
    print(c("\n  ── Predicted vs Actual Symbol Distribution ───────", D+W))
    nn.forward(X_val)
    for r in range(NUM_REEL):
        probs    = nn.out[r].mean(axis=0)
        actuals  = np.bincount(Y_val[:, r], minlength=NUM_SYM) / len(Y_val)
        true_probs = np.array(WEIGHTS) / sum(WEIGHTS)
        print(f"\n  Reel {r+1}:")
        print(f"  {'Symbol':6}  {'Predicted':>10}  {'Actual':>8}  {'True Prob':>9}  Chart")
        print(c("  " + "─"*60, D))
        for s in range(NUM_SYM):
            sym  = SYMBOLS[s]
            col  = SYM_COLS[s]
            diff = abs(probs[s] - true_probs[s])
            dc   = G if diff < 0.01 else Y if diff < 0.03 else R
            print(f"  {c(sym,col):8}  {probs[s]:>10.3f}  {actuals[s]:>8.3f}  "
                  f"{c(f'{true_probs[s]:.3f}',dc):>19}  {bar(probs[s])}")

def live_predict(nn, history_buf):
    """Interactive prediction loop."""
    print(c("\n  ── Live Prediction Mode ──────────────────────────", D+W))
    print(c("  Press Enter to spin | q + Enter to quit\n", D))

    while True:
        cmd = input(c("  > ", C)).strip().lower()
        if cmd == "q":
            break

        # build input from history buffer
        x = np.zeros(WINDOW * NUM_REEL * NUM_SYM)
        for t, s in enumerate(history_buf[-WINDOW:]):
            for r, sym in enumerate(s):
                x[t*NUM_REEL*NUM_SYM + r*NUM_SYM + sym] = 1.0

        predictions = nn.predict_single(x)
        actual      = spin()

        print(f"  {'':6}  {'Pred':>12}  {'Conf':>7}  {'Actual':>10}  {'✓/✗':>4}")
        print(c("  " + "─"*50, D))
        for r in range(NUM_REEL):
            ps, conf  = predictions[r]
            asym      = actual[r]
            ok        = "✓" if ps == asym else "✗"
            ok_col    = G if ps == asym else R
            print(f"  Reel {r+1}  "
                  f"{c(SYMBOLS[ps], SYM_COLS[ps]):>22}  "
                  f"{conf*100:>6.1f}%  "
                  f"{c(SYMBOLS[asym], SYM_COLS[asym]):>20}  "
                  f"{c(ok, ok_col)}")

        history_buf.append(actual)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    random.seed(42)
    os.system("cls" if os.name == "nt" else "clear")
    print_header()

    # 1. Generate data
    print(c("  [1/4] Generating spin history …", W))
    X_all, Y_all = generate_data(6000)
    split = int(0.8 * len(X_all))
    X_train, Y_train = X_all[:split], Y_all[:split]
    X_val,   Y_val   = X_all[split:], Y_all[split:]
    print(f"       Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")

    # 2. Build model
    print(c("\n  [2/4] Building network …", W))
    input_dim = WINDOW * NUM_REEL * NUM_SYM
    nn = NeuralNetwork(input_dim, hidden1=256, hidden2=128, lr=0.002)
    params = (input_dim*256 + 256 + 256*128 + 128 + 128*NUM_SYM*NUM_REEL + NUM_SYM*NUM_REEL)
    print(f"       Input → 256 → 128 → (7×3)   |  Parameters: {params:,}")

    # 3. Train
    print(c("\n  [3/4] Training …\n", W))
    t0      = time.time()
    history = train(nn, X_train, Y_train, X_val, Y_val,
                    epochs=50, batch=128, patience=10)
    elapsed = time.time() - t0
    print(f"\n       Finished in {elapsed:.1f}s")

    # 4. Evaluate
    print(c("\n  [4/4] Evaluation\n", W))
    nn.forward(X_val)
    final_acc  = nn.accuracy(Y_val)
    final_loss = nn.loss(Y_val)
    baseline   = 1 / NUM_SYM            # random guess accuracy
    true_max   = max(w/sum(WEIGHTS) for w in WEIGHTS)   # best-guess always picks most common

    print(f"  Final val accuracy : {c(f'{final_acc*100:.2f}%', G)}")
    print(f"  Final val loss     : {c(f'{final_loss:.4f}', Y)}")
    print(f"  Random baseline    : {c(f'{baseline*100:.2f}%', R)}")
    print(f"  Always-most-common : {c(f'{true_max*100:.2f}%', Y)}")
    lift = final_acc / baseline
    print(f"  Lift over random   : {c(f'{lift:.2f}x', C)}")

    show_history(history)
    show_distribution(nn, X_val, Y_val)

    # 5. Live prediction
    print(c("\n  ══════════════════════════════════════════════════", Y))
    print(c("  Network learned the symbol distribution.", W))
    print(c("  It cannot beat randomness — but watch how confident it is!\n", D))

    # seed history buffer with real spins
    history_buf = [spin() for _ in range(WINDOW)]
    live_predict(nn, history_buf)

    print(c("\n  Done. 🎰🧠\n", BD+W))

if __name__ == "__main__":
    main()
