"""
=============================================================
  Neural Network Classifier for IRIS Dataset
  With MSE Loss + No Reg / LASSO (L1) / Ridge (L2)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 1. PRE-PROCESSING
# ─────────────────────────────────────────────
def normalize(X_train, X_test):
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std

def one_hot(y, num_classes=3):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


# ─────────────────────────────────────────────
# 2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────
def relu(z):         return np.maximum(0, z)
def relu_deriv(z):   return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
# 3. NEURAL NETWORK CLASS
# ─────────────────────────────────────────────
class NeuralNetwork:
    def __init__(self, reg_type='none', lam=0.001, lr=0.01,
                 epochs=1000, batch_size=16, seed=42):
        self.reg_type       = reg_type
        self.lam            = lam
        self.lr             = lr
        self.epochs         = epochs
        self.batch_size     = batch_size
        np.random.seed(seed)
        self._init_weights()
        self.loss_history      = []
        self.train_acc_history = []
        self.test_acc_history  = []

    def _init_weights(self):
        def xavier(fan_in, fan_out):
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        self.W1 = xavier(4,  16);  self.b1 = np.zeros((1, 16))
        self.W2 = xavier(16,  8);  self.b2 = np.zeros((1,  8))
        self.W3 = xavier(8,   3);  self.b3 = np.zeros((1,  3))

    def forward(self, X):
        self.Z1 = X       @ self.W1 + self.b1;  self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2;  self.A2 = relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3;  self.A3 = softmax(self.Z3)
        return self.A3

    def compute_loss(self, y_pred, y_true):
        mse     = np.mean((y_pred - y_true) ** 2)
        weights = [self.W1, self.W2, self.W3]
        if self.reg_type == 'l2':
            penalty = self.lam * sum(np.sum(w**2)       for w in weights)
        elif self.reg_type == 'l1':
            penalty = self.lam * sum(np.sum(np.abs(w))  for w in weights)
        else:
            penalty = 0.0
        return mse + penalty

    def backward(self, X, y_true):
        n   = X.shape[0]
        dA3 = 2 * (self.A3 - y_true) / n

        dW3 = self.A2.T @ dA3
        db3 = dA3.sum(axis=0, keepdims=True)
        dA2 = dA3 @ self.W3.T * relu_deriv(self.Z2)

        dW2 = self.A1.T @ dA2
        db2 = dA2.sum(axis=0, keepdims=True)
        dA1 = dA2 @ self.W2.T * relu_deriv(self.Z1)

        dW1 = X.T @ dA1
        db1 = dA1.sum(axis=0, keepdims=True)

        if self.reg_type == 'l2':
            dW3 += 2 * self.lam * self.W3
            dW2 += 2 * self.lam * self.W2
            dW1 += 2 * self.lam * self.W1
        elif self.reg_type == 'l1':
            dW3 += self.lam * np.sign(self.W3)
            dW2 += self.lam * np.sign(self.W2)
            dW1 += self.lam * np.sign(self.W1)

        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def fit(self, X, y_oh, X_test=None, y_test=None):
        n        = X.shape[0]
        y_labels = np.argmax(y_oh, axis=1)
        for epoch in range(self.epochs):
            idx  = np.random.permutation(n)
            X_s, y_s = X[idx], y_oh[idx]
            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start+self.batch_size]
                yb = y_s[start:start+self.batch_size]
                self.forward(Xb)
                self.backward(Xb, yb)
            if epoch % 10 == 0:
                pred = self.forward(X)
                loss = self.compute_loss(pred, y_oh)
                self.loss_history.append(loss)
                self.train_acc_history.append(self.accuracy(X, y_labels))
                if X_test is not None and y_test is not None:
                    self.test_acc_history.append(self.accuracy(X_test, y_test))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100

    def mse_score(self, X, y):
        pred = self.forward(X)
        y_oh = one_hot(y)
        return np.mean((pred - y_oh) ** 2)


# ─────────────────────────────────────────────
# 4. CONFUSION MATRIX
# ─────────────────────────────────────────────
def confusion_matrix_manual(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ─────────────────────────────────────────────
# 5. CONSTANTS
# ─────────────────────────────────────────────
COLORS = {
    'none': '#2196F3',
    'l1'  : '#FF9800',
    'l2'  : '#4CAF50',
}
LABELS = {
    'none': 'No Regularization',
    'l1'  : 'LASSO (L1)  λ=0.001',
    'l2'  : 'Ridge (L2)  λ=0.0005',
}
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']


# ─────────────────────────────────────────────
# 6. PLOTTING
# ─────────────────────────────────────────────
def plot_all(models, X_train, y_train, X_test, y_test):
    fig = plt.figure(figsize=(22, 22))
    fig.patch.set_facecolor('#0F1117')
    gs  = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── (A) Loss Curves ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1A1D27')
    for key, model in models.items():
        x_vals = np.arange(len(model.loss_history)) * 10
        ax1.plot(x_vals, model.loss_history,
                 color=COLORS[key], linewidth=2.5, label=LABELS[key])
    ax1.set_title('Training Loss over Epochs',
                  color='white', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', color='#AAAAAA')
    ax1.set_ylabel('Loss',  color='#AAAAAA')
    ax1.tick_params(colors='#AAAAAA')
    ax1.legend(fontsize=11, facecolor='#1A1D27', labelcolor='white')
    for spine in ax1.spines.values(): spine.set_edgecolor('#333')

    # ── (B) Train Accuracy Curves ─────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor('#1A1D27')
    for key, model in models.items():
        x_vals = np.arange(len(model.train_acc_history)) * 10
        ax2.plot(x_vals, model.train_acc_history,
                 color=COLORS[key], linewidth=2.5, label=LABELS[key])
    ax2.set_title('Train Accuracy over Epochs',
                  color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', color='#AAAAAA')
    ax2.set_ylabel('Accuracy (%)', color='#AAAAAA')
    ax2.tick_params(colors='#AAAAAA')
    ax2.legend(fontsize=10, facecolor='#1A1D27', labelcolor='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#333')

    # ── (C) Test Accuracy Curves ──────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor('#1A1D27')
    for key, model in models.items():
        x_vals = np.arange(len(model.test_acc_history)) * 10
        ax3.plot(x_vals, model.test_acc_history,
                 color=COLORS[key], linewidth=2.5, label=LABELS[key])
    ax3.set_title('Test Accuracy over Epochs',
                  color='white', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch', color='#AAAAAA')
    ax3.set_ylabel('Accuracy (%)', color='#AAAAAA')
    ax3.tick_params(colors='#AAAAAA')
    ax3.legend(fontsize=10, facecolor='#1A1D27', labelcolor='white')
    for spine in ax3.spines.values(): spine.set_edgecolor('#333')

    # ── (D) Confusion Matrices ────────────────────
    for col, (key, model) in enumerate(models.items()):
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor('#1A1D27')
        preds = model.predict(X_test)
        cm    = confusion_matrix_manual(y_test, preds)
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else '#333',
                        fontsize=14, fontweight='bold')
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(CLASS_NAMES, color='#AAAAAA', fontsize=9, rotation=15)
        ax.set_yticklabels(CLASS_NAMES, color='#AAAAAA', fontsize=9)
        ax.set_title(f'Confusion Matrix\n{LABELS[key]}',
                     color=COLORS[key], fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', color='#AAAAAA')
        ax.set_ylabel('Actual',    color='#AAAAAA')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')

    # ── (E) Accuracy Bar Chart ────────────────────
    keys    = list(models.keys())
    accs    = [models[k].accuracy(X_test, y_test)  for k in keys]
    mses    = [models[k].mse_score(X_test, y_test) for k in keys]
    colors  = [COLORS[k] for k in keys]
    xlabels = [LABELS[k] for k in keys]
    x       = np.arange(len(keys))

    ax_acc = fig.add_subplot(gs[3, :2])
    ax_acc.set_facecolor('#1A1D27')
    bars = ax_acc.bar(x, accs, color=colors, width=0.5,
                      edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, accs):
        ax_acc.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center',
                    color='white', fontsize=12, fontweight='bold')
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(xlabels, color='#AAAAAA', fontsize=10)
    ax_acc.set_ylim(0, 115)
    ax_acc.set_ylabel('Accuracy (%)', color='#AAAAAA')
    ax_acc.set_title('Test Accuracy Comparison',
                     color='white', fontsize=13, fontweight='bold')
    ax_acc.tick_params(colors='#AAAAAA')
    for spine in ax_acc.spines.values(): spine.set_edgecolor('#333')

    # ── (F) MSE Bar Chart ─────────────────────────
    ax_mse = fig.add_subplot(gs[3, 2])
    ax_mse.set_facecolor('#1A1D27')
    bars2 = ax_mse.bar(x, mses, color=colors, width=0.5,
                       edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, mses):
        ax_mse.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center',
                    color='white', fontsize=10, fontweight='bold')
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(['No Reg', 'L1', 'L2'], color='#AAAAAA', fontsize=9)
    ax_mse.set_ylabel('MSE', color='#AAAAAA')
    ax_mse.set_title('Test MSE Comparison',
                     color='white', fontsize=13, fontweight='bold')
    ax_mse.tick_params(colors='#AAAAAA')
    for spine in ax_mse.spines.values(): spine.set_edgecolor('#333')

    fig.suptitle('IRIS Neural Network Classifier — Results Dashboard',
                 color='white', fontsize=17, fontweight='bold', y=0.99)

    plt.savefig('iris_results.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("✅ Plot saved as iris_results.png")


# ─────────────────────────────────────────────
# 7. PRINT REPORT
# ─────────────────────────────────────────────
def print_report(models, X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("         IRIS NEURAL NETWORK — RESULTS REPORT")
    print("="*60)
    print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Test MSE':>12}")
    print("-"*60)
    for key, model in models.items():
        tr_acc = model.accuracy(X_train, y_train)
        te_acc = model.accuracy(X_test,  y_test)
        te_mse = model.mse_score(X_test,  y_test)
        print(f"{LABELS[key]:<25} {tr_acc:>9.2f}% {te_acc:>9.2f}% {te_mse:>12.6f}")
    print("="*60)

    print("\n📊 Confusion Matrices (Test Set):")
    for key, model in models.items():
        preds = model.predict(X_test)
        cm    = confusion_matrix_manual(y_test, preds)
        print(f"\n  ── {LABELS[key]} ──")
        print(f"  {'':10}", end="")
        for c in CLASS_NAMES: print(f"  {c:>10}", end="")
        print()
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i]:>10}", end="")
            for val in row: print(f"  {val:>10}", end="")
            print()


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("📂 Loading data...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔧 Normalizing features...")
    X_train_n, X_test_n = normalize(X_train, X_test)
    y_train_oh = one_hot(y_train)

    configs = {
        'none': dict(reg_type='none', lam=0.0),
        'l1'  : dict(reg_type='l1',  lam=0.001),
        'l2'  : dict(reg_type='l2',  lam=0.0005),
    }

    models = {}
    for key, cfg in configs.items():
        print(f"\n🚀 Training: {LABELS[key]} ...")
        model = NeuralNetwork(**cfg, lr=0.05, epochs=2000, batch_size=16)
        model.fit(X_train_n, y_train_oh, X_test_n, y_test)
        models[key] = model
        acc = model.accuracy(X_test_n, y_test)
        print(f"   ✅ Done — Test Accuracy: {acc:.1f}%")

    print_report(models, X_train_n, y_train, X_test_n, y_test)

    print("\n📊 Generating plots...")
    plot_all(models, X_train_n, y_train, X_test_n, y_test)
    print("\n🎉 All done!")
    