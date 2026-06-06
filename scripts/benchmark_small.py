"""小型基准：核预计算与 AMP 效果对比

运行方式：
    python scripts/benchmark_small.py

输出：控制台打印各项耗时（秒）。脚本尽量在 CPU/GPU 自动适配下运行。
"""
import time
import os
import sys
import argparse
import statistics

# Ensure project root is on sys.path so `core` can be imported when running this script directly
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_kernels
from sklearn.kernel_approximation import Nystroem, RBFSampler

try:
    from core.deep_learning import TorchMLP
except Exception:
    TorchMLP = None

from core.kernel_cache import KernelCache


def _create_kernel_approximation(X_tr, X_val, orig_kernel, gamma=1.0, degree=3, coef0=1.0, n_components=512):
    if orig_kernel == 'rbf':
        transformer = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    else:
        transformer = Nystroem(kernel=orig_kernel, gamma=gamma, degree=degree, coef0=coef0,
                               n_components=n_components, random_state=42)
    X_tr_approx = transformer.fit_transform(X_tr)
    X_val_approx = transformer.transform(X_val)
    return X_tr_approx, X_val_approx


def bench_svm(n_samples=3000, n_features=36, repeat=1, use_approx=False):
    print("\n=== SVM kernel precompute benchmark ===")
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    results = {'direct': [], 'precompute_kernel': [], 'precomputed': [], 'approx_transform': [], 'approx': []}

    for _ in range(repeat):
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', tol=1e-3, max_iter=1000)
        t0 = time.time()
        clf.fit(X, y)
        results['direct'].append(time.time() - t0)

        t0 = time.time()
        K = pairwise_kernels(X, X, metric='rbf', n_jobs=min(os.cpu_count() or 1, 8))
        results['precompute_kernel'].append(time.time() - t0)

        clf2 = SVC(kernel='precomputed', C=1.0, tol=1e-3, max_iter=1000)
        t0 = time.time()
        clf2.fit(K, y)
        results['precomputed'].append(time.time() - t0)

        if use_approx:
            n_components = min(512, max(128, X.shape[0] // 8))
            t0 = time.time()
            X_tr_approx, X_val_approx = _create_kernel_approximation(X, X, 'rbf', gamma=1.0, n_components=n_components)
            results['approx_transform'].append(time.time() - t0)

            clf3 = SVC(kernel='linear', C=1.0, tol=1e-3, max_iter=1000)
            t0 = time.time()
            clf3.fit(X_tr_approx, y)
            results['approx'].append(time.time() - t0)

            acc = accuracy_score(y, clf3.predict(X_val_approx))
            print(f"Approx features acc: {acc:.4f}")

    print(f"Direct SVC.fit avg: {statistics.mean(results['direct']):.3f}s")
    print(f"pairwise_kernels avg: {statistics.mean(results['precompute_kernel']):.3f}s")
    print(f"Precomputed-kernel SVC.fit avg: {statistics.mean(results['precomputed']):.3f}s")
    if use_approx:
        print(f"Approx transform avg: {statistics.mean(results['approx_transform']):.3f}s")
        print(f"Approx linear SVC.fit avg: {statistics.mean(results['approx']):.3f}s")

    cache = KernelCache()
    key = ('rbf', X.shape, 0)
    cache.set(key, K)
    t_set_start = time.time()
    cache.set(key, K)
    t_set = time.time() - t_set_start
    t_get_start = time.time()
    K_cached = cache.get(key)
    t_get = time.time() - t_get_start
    print(f"Cache set/get times: set {t_set:.4f}s get {t_get:.4f}s (cached shape {None if K_cached is None else K_cached.shape})")


def bench_torchmlp(n_samples=5000, n_features=36, repeat=1):
    print("\n=== TorchMLP AMP benchmark ===")
    if TorchMLP is None:
        print("TorchMLP not available (PyTorch not installed). Skipping.")
        return

    # Ensure DataLoader uses num_workers=0 in this benchmark to avoid Windows multiprocessing hangs
    try:
        orig_create = TorchMLP._create_dataloader

        def _create_dataloader_zero(self, X, y, shuffle: bool = True, device=None):
            dl = orig_create(self, X, y, shuffle=shuffle, device=device)
            try:
                nw = getattr(dl, 'num_workers', 0)
            except Exception:
                nw = 0
            if nw and hasattr(dl, 'dataset'):
                from torch.utils.data import DataLoader
                return DataLoader(dl.dataset, batch_size=dl.batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, persistent_workers=False)
            return dl

        TorchMLP._create_dataloader = _create_dataloader_zero
    except Exception:
        pass

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=min(20, n_features), random_state=42)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    times = {'amp_off': [], 'amp_on': []}
    for use_amp in (False, True):
        for _ in range(repeat):
            model = TorchMLP(task_type='classification', epochs=5, batch_size=256, use_amp=use_amp, verbose=False, device=device)
            t0 = time.time()
            model.fit(X, y)
            times['amp_on' if use_amp else 'amp_off'].append(time.time() - t0)

        avg_time = statistics.mean(times['amp_on' if use_amp else 'amp_off'])
        print(f"TorchMLP use_amp={use_amp} avg time over {repeat} runs: {avg_time:.3f}s")

    return times


def main():
    parser = argparse.ArgumentParser(description='Small benchmark for kernel precompute and TorchMLP AMP.')
    parser.add_argument('--svm-samples', type=int, default=3000, help='Number of samples for SVM benchmark')
    parser.add_argument('--svm-features', type=int, default=36, help='Number of features for SVM benchmark')
    parser.add_argument('--svm-repeat', type=int, default=1, help='Repeat count for SVM benchmark')
    parser.add_argument('--svm-approx', action='store_true', help='Also benchmark kernel approximation with linear SVC')
    parser.add_argument('--torch-samples', type=int, default=5000, help='Number of samples for TorchMLP benchmark')
    parser.add_argument('--torch-features', type=int, default=36, help='Number of features for TorchMLP benchmark')
    parser.add_argument('--torch-repeat', type=int, default=1, help='Repeat count for TorchMLP benchmark')
    args = parser.parse_args()

    print("Running small benchmarks (sample sizes kept moderate).")
    bench_svm(n_samples=args.svm_samples, n_features=args.svm_features, repeat=args.svm_repeat, use_approx=args.svm_approx)
    bench_torchmlp(n_samples=args.torch_samples, n_features=args.torch_features, repeat=args.torch_repeat)


if __name__ == '__main__':
    main()
