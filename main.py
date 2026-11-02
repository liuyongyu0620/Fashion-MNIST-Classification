import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = np.array([0, 2, 4, 6], dtype=int)

def read_csv(path, labeled):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    # Compatible with Lab 2: first row is a header.
    data = np.asarray(rows[1:], dtype=np.float32)
    expected = 785 if labeled else 784
    if data.ndim != 2 or data.shape[1] != expected:
        raise ValueError(f'{path}: expected {expected} columns, got {data.shape}')
    if labeled:
        labels = data[:, 0].astype(int)
        if not np.all(np.isin(labels, LABELS)):
            raise ValueError('Expected Group B labels: 0, 2, 4, 6')
        return (data[:, 1:] / 255.0).astype(np.float64), np.searchsorted(LABELS, labels)
    return (data / 255.0).astype(np.float64)

def split_data(X, y, seed=42, valid_ratio=0.2):
    rng = np.random.default_rng(seed)
    # Stratified split to preserve all four classes.
    train_idx, valid_idx = [], []
    for c in range(4):
        ids = rng.permutation(np.flatnonzero(y == c))
        n_valid = max(1, int(round(len(ids) * valid_ratio)))
        valid_idx.extend(ids[:n_valid]); train_idx.extend(ids[n_valid:])
    return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]

def activate(z, kind):
    if kind == 'relu': return np.maximum(0, z)
    if kind == 'sigmoid':
        return np.exp(-np.logaddexp(0, -z))
    if kind == 'tanh': return np.tanh(z)
    raise ValueError(kind)

def derivative(z, a, kind):
    if kind == 'relu': return (z > 0).astype(z.dtype)
    if kind == 'sigmoid': return a * (1 - a)
    if kind == 'tanh': return 1 - a*a
    raise ValueError(kind)

def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, sizes, activation='relu', seed=42):
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for layer, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            # He for ReLU hidden layers, Xavier for Sigmoid/Tanh and output.
            scale = np.sqrt(2 / n_in) if layer < len(sizes)-2 and activation == 'relu' else np.sqrt(2 / (n_in + n_out))
            self.W.append(rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = a @ W + b
            zs.append(z)
            a = activate(z, self.activation)
            activations.append(a)
        logits = a @ self.W[-1] + self.b[-1]
        zs.append(logits)
        activations.append(softmax(logits))
        return activations, zs

    def loss_and_accuracy(self, X, y):
        p = self.forward(X)[0][-1]
        loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
        return float(loss), float((p.argmax(axis=1) == y).mean())

    def train_batch(self, X, y, lr):
        acts, zs = self.forward(X)
        n = len(y)
        delta = acts[-1].copy()
        delta[np.arange(n), y] -= 1
        delta /= n
        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.b)
        for layer in reversed(range(len(self.W))):
            grad_W[layer] = acts[layer].T @ delta
            grad_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.W[layer].T) * derivative(zs[layer-1], acts[layer], self.activation)
        for i in range(len(self.W)):
            self.W[i] -= lr * grad_W[i]
            self.b[i] -= lr * grad_b[i]

def train(model, Xtr, ytr, Xval, yval, lr, batch_size, epochs, patience, seed):
    rng = np.random.default_rng(seed)
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    best_loss = np.inf; best_epoch = 0; stale = 0
    best_weights = None
    for epoch in range(1, epochs+1):
        for ids in np.array_split(rng.permutation(len(ytr)), range(batch_size, len(ytr), batch_size)):
            model.train_batch(Xtr[ids], ytr[ids], lr)
        tr_loss, tr_acc = model.loss_and_accuracy(Xtr, ytr)
        val_loss, val_acc = model.loss_and_accuracy(Xval, yval)
        for key, value in [('train_loss', tr_loss), ('valid_loss', val_loss), ('train_acc', tr_acc), ('valid_acc', val_acc)]:
            history[key].append(value)
        print(f'Epoch {epoch:3d}: train loss={tr_loss:.4f}, acc={tr_acc:.4f}; valid loss={val_loss:.4f}, acc={val_acc:.4f}')
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss; best_epoch = epoch; stale = 0
            best_weights = ([w.copy() for w in model.W], [b.copy() for b in model.b])
        else:
            stale += 1
            if stale >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    model.W, model.b = best_weights
    return history, best_epoch

def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('loss', 'acc')):
        ax.plot(history['train_'+metric], label='Training')
        ax.plot(history['valid_'+metric], label='Validation')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy' if metric=='acc' else 'Cross-entropy loss')
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    X, y = read_csv(args.train, True)
    Xt = read_csv(args.test, False)
    Xtr, ytr, Xval, yval = split_data(X, y, args.seed)
    summaries = []
    for activation in args.activations:
        model = MLP([784, *args.hidden, 4], activation, args.seed)
        print(f'\nTraining activation={activation}, architecture={[784,*args.hidden,4]}')
        history, best_epoch = train(model, Xtr, ytr, Xval, yval, args.lr, args.batch_size, args.epochs, args.patience, args.seed)
        tr_loss, tr_acc = model.loss_and_accuracy(Xtr, ytr)
        val_loss, val_acc = model.loss_and_accuracy(Xval, yval)
        preds = LABELS[model.forward(Xt)[0][-1].argmax(axis=1)]
        np.savetxt(out / f'predictions_{activation}.csv', preds, fmt='%d', delimiter=',', header='label', comments='')
        plot_history(history, out / f'curves_{activation}.png')
        np.savez(out / f'model_{activation}.npz', **{f'W{i}':w for i,w in enumerate(model.W)}, **{f'b{i}':b for i,b in enumerate(model.b)})
        summary = {'activation': activation, 'architecture': [784,*args.hidden,4], 'learning_rate': args.lr, 'batch_size': args.batch_size, 'max_epochs': args.epochs, 'epochs_run': len(history['train_loss']), 'best_epoch': best_epoch, 'train_loss': tr_loss, 'train_accuracy': tr_acc, 'validation_loss': val_loss, 'validation_accuracy': val_acc, 'prediction_file': f'predictions_{activation}.csv'}
        summaries.append(summary)
        (out / f'history_{activation}.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    (out / 'comparison.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(f'\nSaved outputs to {out.resolve()}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Lab 3 MLP from scratch (NumPy only)')
    p.add_argument('--train', default='Group_B_train.csv')
    p.add_argument('--test', default='Group_B_test.csv')
    p.add_argument('--output', default='lab3_results')
    p.add_argument('--hidden', type=int, nargs='+', default=[128], help='1 or 2 hidden layer sizes')
    p.add_argument('--activations', nargs='+', choices=['relu','sigmoid','tanh'], default=['relu','sigmoid'])
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if len(args.hidden) not in (1,2) or any(h <= 0 for h in args.hidden): p.error('Specify 1 or 2 positive hidden layer sizes')
    if args.lr <= 0 or args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0: p.error('Training parameters must be positive')
    run(args)
