"""Memory-optimized S2D-GRL using H feature-column blocks."""

import math
import sys
import time

import torch


def _get_process_peak_memory_mb():
    """Process-lifetime peak RAM in MiB, using only the standard library."""
    try:
        if sys.platform == 'win32':
            import ctypes

            class MemoryCounters(ctypes.Structure):
                _fields_ = [
                    ('cb', ctypes.c_ulong),
                    ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]

            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.argtypes = []
            get_process.restype = ctypes.c_void_p
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
            get_memory.restype = ctypes.c_int
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PeakWorkingSetSize / (1024 ** 2)

        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 ** 2 if sys.platform == 'darwin' else 1024)
    except (ImportError, AttributeError, OSError):
        return None


class DBP_GRL:
    def __init__(
        self,
        alpha_S,
        alpha_A,
        beta_S,
        beta_A,
        tau_S,
        tau_A,
        epoch=100,
        block_num=20,
    ):
        if isinstance(block_num, bool) or not isinstance(block_num, int):
            raise TypeError('block_num must be an integer')
        if block_num < 1:
            raise ValueError('block_num must be at least 1')

        self.alpha_S = alpha_S
        self.alpha_A = alpha_A
        self.beta_S = beta_S
        self.beta_A = beta_A
        self.tau_S = tau_S
        self.tau_A = tau_A
        self.epoch = epoch
        self.block_num = block_num

    def _feature_blocks(self, feature_dim):
        block_num = min(self.block_num, feature_dim)
        block_size = math.ceil(feature_dim / block_num)
        for start in range(0, feature_dim, block_size):
            yield start, min(start + block_size, feature_dim)

    @staticmethod
    def _normalize_rows_inplace(H, eps=1e-12):
        denominator = torch.linalg.vector_norm(H, ord=2, dim=1, keepdim=True)
        denominator.clamp_min_(eps)
        H.div_(denominator)
        return H

    def _standardized_normalized_copy(self, H):
        Z = H.clone()
        self._normalize_rows_inplace(Z)
        std, mean = torch.std_mean(Z, dim=0, unbiased=False)
        Z.sub_(mean).div_(std)
        self._normalize_rows_inplace(Z)
        return Z

    def _adj_matmul(self, A, H):
        if self.block_num == 1:
            return A @ H

        output = H.new_empty((A.size(0), H.size(1)))
        for start, end in self._feature_blocks(H.size(1)):
            block_result = A @ H[:, start:end]
            output[:, start:end].copy_(block_result)
            del block_result
        return output

    def _subtract_decorrelation_inplace(self, target, Z, beta):
        """target -= beta * Z @ (Z.T @ Z), without a full N x D temporary."""
        gram = Z.T @ Z
        for start, end in self._feature_blocks(Z.size(1)):
            correction = Z @ gram[:, start:end]
            target[:, start:end].add_(correction, alpha=-beta)
            del correction

    def _project_features_inplace(self, target, X):
        """Replace target with X @ (X.T @ centered(target)) by feature blocks."""
        target_mean = target.mean(dim=0)
        projection = X.T @ target
        projection.sub_(torch.outer(X.sum(dim=0), target_mean))

        for start, end in self._feature_blocks(target.size(1)):
            projected_block = X @ projection[:, start:end]
            target[:, start:end].copy_(projected_block)
            del projected_block

    def fit(self, A, X):
        use_cuda = X.is_cuda
        if use_cuda:
            device = X.device
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()
        H_A = X
        H_S = X
        for ep in range(0, self.epoch + 1):
            with torch.no_grad():
                Z_A = self._standardized_normalized_copy(H_A)
                AI = self._adj_matmul(A, Z_A)
                self._normalize_rows_inplace(AI)
                self._subtract_decorrelation_inplace(AI, Z_A, self.beta_A)
                del Z_A

                self._project_features_inplace(AI, X)
                AI.mul_(self.alpha_A).add_(H_A, alpha=1 - self.alpha_A)
                AI.mul_(self.tau_A).add_(H_S, alpha=1 - self.tau_A)
                H_A = AI
                del AI

                Z_S = self._standardized_normalized_copy(H_S)
                TI = self._adj_matmul(A, Z_S)
                self._subtract_decorrelation_inplace(TI, Z_S, self.beta_S)
                del Z_S

                TI.mul_(self.alpha_S).add_(H_S, alpha=1 - self.alpha_S)
                if ep < self.epoch - 1:
                    TI.mul_(self.tau_S).add_(H_A, alpha=1 - self.tau_S)
                H_S = TI
                del TI

        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed_time = time.perf_counter() - start_time

        if use_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            memory_name = 'peak CUDA memory allocated'
        else:
            peak_memory_mb = _get_process_peak_memory_mb()
            memory_name = 'peak process memory'

        print(f'Total training time: {elapsed_time:.4f} s')
        if peak_memory_mb is None:
            print(f'{memory_name}: unavailable')
        else:
            print(f'{memory_name}: {peak_memory_mb:.2f} MiB')
            print('---------------test---------------------')
        return H_S



