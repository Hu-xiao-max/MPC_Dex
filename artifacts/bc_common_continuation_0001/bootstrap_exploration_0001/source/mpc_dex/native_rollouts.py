"""Optional compiled exact transition loop for :mod:`mpc_dex.rollouts`.

``NativeMujocoRolloutEvaluator`` retains the Python evaluator's scoring and
thread ownership, replacing only its per-candidate transition loop. Compilation
is opt-in through this class; the default evaluator is untouched. Missing build
tools, unsupported platforms/ABI, callbacks and plugins use its exact Python
fallback unless ``require_native=True``. A failed native call is never retried.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from .rollouts import MujocoRolloutEvaluator


NATIVE_ABI_VERSION = 2


class NativeBackendUnavailable(RuntimeError):
    """The optional native backend cannot safely run in this environment."""


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _callbacks_active():
    return any(getattr(mujoco, name)() is not None
               for name in dir(mujoco) if name.startswith('get_mjcb_'))


def _load_library(cache_directory=None):
    if platform.system() != 'Linux' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise NativeBackendUnavailable('Native loop currently supports 64-bit Linux only')
    package = Path(mujoco.__file__).resolve().parent
    source = Path(__file__).with_name('native') / 'exact_rollout.cc'
    library = package / f'libmujoco.so.{mujoco.__version__}'
    include = package / 'include'
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if not compiler or not library.is_file() or not (include / 'mujoco/mujoco.h').is_file():
        raise NativeBackendUnavailable('MuJoCo headers/library or C++ compiler unavailable')
    try:
        compiler_version = subprocess.run([compiler, '--version'], check=True,
            capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        header_hash = hashlib.sha256()
        for path in sorted(include.rglob('*.h')):
            header_hash.update(str(path.relative_to(include)).encode())
            header_hash.update(path.read_bytes())
        flags = ['-O3', '-std=c++17', '-fPIC', '-shared', '-fno-fast-math', '-ffp-contract=off']
        provenance = {'source_sha256': _digest(source), 'mujoco_version': mujoco.__version__,
            'transition_abi_version': NATIVE_ABI_VERSION,
            'mujoco_runtime_version': mujoco.mj_version(), 'mujoco_library': str(library),
            'mujoco_library_sha256': _digest(library), 'headers_sha256': header_hash.hexdigest(),
            'compiler': str(Path(compiler).resolve()), 'compiler_version': compiler_version,
            'compiler_flags': flags, 'platform': platform.platform(), 'machine': platform.machine()}
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_directory) if cache_directory else (
            Path(__file__).resolve().parents[1] / 'artifacts' / 'native_rollouts')
        destination = cache / key
        destination.mkdir(parents=True, exist_ok=True)
        binary = destination / 'exact_rollout.so'
        if not binary.is_file():
            # A unique build file plus atomic replacement tolerates concurrent
            # evaluators compiling the identical source into the same cache.
            with tempfile.TemporaryDirectory(dir=destination) as temporary:
                build = Path(temporary) / binary.name
                command = [compiler, *flags, f'-I{include}', str(source), str(library),
                           f'-Wl,-rpath,{package}', '-o', str(build)]
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
                os.replace(build, binary)
        provenance.update(build_key=key, binary=str(binary), binary_sha256=_digest(binary))
        (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        loaded = ctypes.CDLL(str(binary))  # CDLL releases the GIL during C calls.
    except (OSError, subprocess.SubprocessError) as error:
        raise NativeBackendUnavailable(f'Native build/load unavailable: {error}') from error
    for name in ('exact_rollout_header_version', 'exact_rollout_library_version',
                 'exact_rollout_number_size', 'exact_rollout_abi_version'):
        function = getattr(loaded, name)
        function.argtypes = []
        function.restype = ctypes.c_int
    if (loaded.exact_rollout_header_version() != mujoco.mj_version() or
        loaded.exact_rollout_library_version() != mujoco.mj_version() or
        loaded.exact_rollout_abi_version() != NATIVE_ABI_VERSION or
        loaded.exact_rollout_number_size() != np.dtype(np.float64).itemsize):
        raise NativeBackendUnavailable('MuJoCo header/runtime/numeric ABI mismatch')
    ptr = ctypes.c_void_p
    integer = ctypes.c_int64
    number = ctypes.c_double
    loaded.exact_rollout_batch.argtypes = [
        ptr, ptr, ptr, ptr, integer, integer, integer, ptr, integer,
        ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr,
        number, number, number, number, ctypes.c_int, ctypes.c_int, number,
        ctypes.c_int, ptr, ptr, ptr, ptr]
    loaded.exact_rollout_batch.restype = ctypes.c_int
    return loaded, provenance


def _owner_pointer(owner, expected_type, model=None):
    # No integer pointer API is exposed to callers. These are read-only addresses
    # supplied by MuJoCo's own typed Python objects, whose strong references live
    # at least as long as the synchronous native call.
    if type(owner) is not expected_type:
        raise NativeBackendUnavailable(f'Expected exact {expected_type.__name__} owner')
    if model is not None and owner.model is not model:
        raise NativeBackendUnavailable('mjData is owned by a different mjModel')
    address = getattr(owner, '_address', None)
    if not isinstance(address, int) or address <= 0 or address % ctypes.alignment(ctypes.c_void_p):
        raise NativeBackendUnavailable('MuJoCo does not expose a valid aligned owner address')
    return address


def _array_pointer(array, shape, dtype, *, writable=False):
    if (not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.dtype(dtype)
            or not array.flags.c_contiguous or not array.flags.aligned
            or (writable and not array.flags.writeable)):
        raise ValueError(f'Expected aligned contiguous {np.dtype(dtype)} array with shape {shape}')
    return array.ctypes.data


class NativeMujocoRolloutEvaluator(MujocoRolloutEvaluator):
    """Drop-in exact evaluator with optional native transitions and safe fallback.

    Like the base evaluator, one instance is used by one planning thread, while
    its private worker pool parallelizes candidates. The environment/model must
    not be mutated concurrently with planning. ``native_provenance`` records
    compiler, source, header, library and binary hashes for each compiled build.
    """

    def __init__(self, env, timestep_seconds=.01, config=None, *, require_native=False,
                 cache_directory=None):
        self.native_library = None
        self.native_provenance = None
        self.native_unavailable_reason = None
        self.require_native = require_native
        super().__init__(env, timestep_seconds, config)
        try:
            _owner_pointer(env.model, mujoco.MjModel)
            _owner_pointer(env.data, mujoco.MjData, env.model)
            if env.model.nplugin or _callbacks_active():
                raise NativeBackendUnavailable('Native transitions require no plugins or global callbacks')
            if np.dtype(int) != np.dtype(np.int64):
                raise NativeBackendUnavailable('Native step counts require 64-bit NumPy integers')
            self.native_library, self.native_provenance = _load_library(cache_directory)
        except NativeBackendUnavailable as error:
            self.native_unavailable_reason = str(error)
            if require_native:
                self.close()
                raise

    def __call__(self, sequences):
        # Match base validation while accepting its strided ndarray inputs.
        sequences = np.ascontiguousarray(sequences, dtype=np.float64)
        self._use_native = self.native_library is not None and not _callbacks_active() and not self.env.model.nplugin
        if self.native_library is not None and not self._use_native:
            self.native_unavailable_reason = 'Plugins or global callbacks were activated after initialization'
            if self.require_native:
                raise NativeBackendUnavailable(self.native_unavailable_reason)
        self._boundary_fallbacks = [0] * self.workers
        result = super().__call__(sequences)
        self.last_diagnostics['backend'] = 'exact_native_step_threads' if self._use_native else 'exact_step_threads'
        if self._use_native:
            self.last_diagnostics['native_build'] = self.native_provenance
            self.last_diagnostics['native_boundary_fallback_candidates'] = sum(self._boundary_fallbacks)
        else:
            self.last_diagnostics['native_unavailable_reason'] = self.native_unavailable_reason
        return result

    def _simulate_candidates(self, worker, candidates, sequences, snapshot,
                             initial_contactless, states, finger_counts, step_counts, reasons):
        if not self._use_native:
            return super()._simulate_candidates(worker, candidates, sequences, snapshot,
                initial_contactless, states, finger_counts, step_counts, reasons)
        env, model = self.env, self.env.model
        data = self.worker_data[worker]
        n, horizon, actuators = sequences.shape
        steps = horizon * self.repeat
        size = mujoco.mj_stateSize(model, self.state_spec)
        if actuators != model.nu or len(reasons) != n:
            raise ValueError('Native actuator/reason dimensions disagree with the model')
        selected = np.ascontiguousarray(list(candidates), dtype=np.int64)
        if not len(selected) or np.any(selected < 0) or np.any(selected >= n) or len(np.unique(selected)) != len(selected):
            raise ValueError('Candidate indices must be nonempty, unique and in bounds')
        initial_position = np.ascontiguousarray(env.initial_position, dtype=np.float64)
        numeric = [env.config.timestep, initial_contactless, env.config.contact_loss_seconds,
                   self.config.workspace_radius, env.config.palm_support_force_tolerance]
        if (not np.isfinite(numeric).all() or not np.isfinite(initial_position).all()
                or min(numeric) < 0 or numeric[0] <= 0 or numeric[2] <= 0):
            raise ValueError('Native rejection configuration must be finite and nonnegative')
        # Each worker owns its reason buffer; the shared trajectory rows are
        # disjoint by the base evaluator's range partition.
        codes = np.zeros(n, dtype=np.uint8)
        status = self.native_library.exact_rollout_batch(
            _owner_pointer(model, mujoco.MjModel), _owner_pointer(data, mujoco.MjData, model),
            _owner_pointer(snapshot, mujoco.MjData, model),
            _array_pointer(sequences, (n, horizon, actuators), np.float64), n, horizon, self.repeat,
            _array_pointer(selected, (len(selected),), np.int64), len(selected),
            _array_pointer(self._finger_masks, (model.ngeom,), np.uint8),
            env.object_geom, env.table_geom, env.object_id, env.palm_id,
            _array_pointer(initial_position, (3,), np.float64), *numeric[:4],
            int(env.config.task == 'rotation'), int(self._initial_palm_support_violation), numeric[4], size,
            _array_pointer(states, (n, steps, size), np.float64, writable=True),
            _array_pointer(finger_counts, (n, steps), np.uint8, writable=True),
            _array_pointer(step_counts, (n,), np.int64, writable=True),
            _array_pointer(codes, (n,), np.uint8, writable=True))
        if status:
            raise RuntimeError(f'Native rollout argument check failed with code {status}')
        names = [None, 'divergence', 'table_contact', 'workspace', 'contact_loss', None, 'palm_support']
        for offset, candidate in enumerate(selected):
            if codes[candidate] == 5:
                initial = np.empty(size, dtype=np.float64)
                mujoco.mj_getState(model, snapshot, initial, self.state_spec)
                states[candidate] = initial
                finger_counts[candidate] = 0
                self._boundary_fallbacks[worker] = len(selected)-offset
                return super()._simulate_candidates(worker, selected[offset:], sequences, snapshot,
                    initial_contactless, states, finger_counts, step_counts, reasons)
            reasons[candidate] = names[codes[candidate]]
