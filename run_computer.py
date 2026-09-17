import numpy as np
import torch
import torch.nn.functional as F
from contextlib import redirect_stdout
from torch_geometric import seed_everything

from DBP_GRL import DBP_GRL
from utils import load, load_index, evaluate_classification


def main():
    # The original downstream evaluator uses cuda:0.
    if not torch.cuda.is_available():
        raise RuntimeError('The existing classification evaluator requires CUDA.')
    device = 'cuda:0'
    seed_everything(0)

    adj, feat, labels, num_class, _, _, _ = load('computer')
    X = feat.to(device)
    F.normalize(X, out=X)
    A = adj.to(device)
    del adj, feat

    model = DBP_GRL(
        alpha_S=1e-1, alpha_A=1e-3,
        beta_S=1e-3, beta_A=1e-3,
        tau_S=0.95, tau_A=0.99,
        epoch=100, block_num=1,
    )
    print('Computer | DBP_GRL | epoch=100 | block_num=1')
    print('Time/memory below: native fit only; excludes data preparation, '
          'final normalization and downstream evaluation. '
          'Peak allocated CUDA memory includes resident A and X.')
    # fit() already synchronizes CUDA and resets/reports peak allocated memory.
    H = model.fit(A, X)
    F.normalize(H, out=H)
    del A, X, model
    torch.cuda.empty_cache()

    if not torch.isfinite(H).all().item():
        raise RuntimeError('Non-finite representations; evaluation stopped.')
    labels = labels.to(device)
    accuracies = []
    for seed in range(10):
        seed_everything(seed)
        train_idx, val_idx, test_idx = load_index('computer')
        # Hide per-epoch prints without changing the original evaluator.
        with redirect_stdout(None):
            _, val_acc, test_acc = evaluate_classification(
                H, num_class, labels, 5e-2, 1e-4,
                train_idx, val_idx, test_idx,
            )
        accuracies.append(float(test_acc))
        print(f'Seed {seed}: val={float(val_acc):.4f}, test={float(test_acc):.4f}')

    print(f'Test accuracy (10 runs): {np.mean(accuracies) * 100:.2f}% '
          f'+/- {np.std(accuracies) * 100:.2f}% (std)')


if __name__ == '__main__':
    main()
