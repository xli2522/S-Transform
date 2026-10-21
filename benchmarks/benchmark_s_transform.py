"""Compare the batched transform with the former row-wise implementation.

Install the optional dependency with ``python -m pip install '.[benchmark]'``
and run this script from the repository root. Each implementation runs in a
fresh process so peak RSS measurements are comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from scipy import fft, signal

from s_transform import s_transform


@dataclass(frozen=True)
class BenchmarkCase:
    """Input shape and options for one benchmark scenario."""

    name: str
    n_samples: int
    signed: bool = False
    output_length: int | None = None


CASES = (
    BenchmarkCase("dense-real", n_samples=4096),
    BenchmarkCase("signed-complex", n_samples=2048, signed=True),
    BenchmarkCase("resampled", n_samples=4096, output_length=2048),
)


def _make_signal(case: BenchmarkCase) -> np.ndarray:
    rng = np.random.default_rng(case.n_samples)
    samples = rng.standard_normal(case.n_samples)
    if case.signed:
        samples = samples + 1j * rng.standard_normal(case.n_samples)
    return samples


def _frequency_indices(case: BenchmarkCase) -> np.ndarray:
    if case.signed:
        signed_indices = np.arange(
            -((case.n_samples - 1) // 2),
            case.n_samples // 2 + 1,
            dtype=np.int64,
        )
    else:
        signed_indices = np.arange(case.n_samples // 2 + 1, dtype=np.int64)
    return np.mod(signed_indices, case.n_samples)


def _rowwise_reference(
    samples: np.ndarray,
    case: BenchmarkCase,
) -> np.ndarray:
    """Run the original one-frequency-at-a-time transform kernel."""
    samples = np.asarray(samples, dtype=np.complex128)
    n_samples = samples.size
    output_length = case.output_length or n_samples
    max_bin = n_samples // 2
    sample_bins = np.arange(n_samples, dtype=np.int64)
    signed_sample_bins = np.where(
        sample_bins <= max_bin,
        sample_bins,
        sample_bins - n_samples,
    )
    spectrum = fft.fft(samples)
    frequency_indices = _frequency_indices(case)
    result = np.empty(
        (frequency_indices.size, output_length),
        dtype=np.complex128,
    )

    for row_index, frequency_index in enumerate(frequency_indices):
        if frequency_index == 0:
            row = np.full(n_samples, spectrum[0] / n_samples, dtype=np.complex128)
        else:
            signed_frequency_index = (
                frequency_index
                if frequency_index <= max_bin
                else frequency_index - n_samples
            )
            shifted = spectrum[
                (frequency_index + np.arange(n_samples, dtype=np.int64))
                % n_samples
            ]
            window = np.exp(
                -2.0
                * np.pi**2
                * (
                    signed_sample_bins
                    / abs(signed_frequency_index)
                )
                ** 2
            )
            row = fft.ifft(shifted * window)
        if output_length != n_samples:
            row = signal.resample(row, output_length)
        result[row_index] = row
    return result


def _run_implementation(
    samples: np.ndarray,
    case: BenchmarkCase,
    implementation: str,
) -> np.ndarray:
    if implementation == "reference":
        return _rowwise_reference(samples, case)

    options: dict[str, Any] = {}
    if case.signed:
        options["frequency_range"] = (-case.n_samples / 2, case.n_samples / 2)
    if case.output_length is not None:
        options["output_length"] = case.output_length
    return s_transform(
        samples,
        sample_rate=float(case.n_samples),
        **options,
    ).data


def _worker(case: BenchmarkCase, implementation: str, repeats: int) -> None:
    samples = _make_signal(case)
    _run_implementation(samples, case, implementation)
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = _run_implementation(samples, case, implementation)
        durations.append(time.perf_counter() - start)

    assert result is not None
    print(
        json.dumps(
            {
                "median_seconds": statistics.median(durations),
                "shape": result.shape,
            }
        )
    )


def _measure_in_subprocess(
    script: Path,
    case: BenchmarkCase,
    implementation: str,
    repeats: int,
) -> tuple[dict[str, Any], int]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--case",
        case.name,
        "--implementation",
        implementation,
        "--repeats",
        str(repeats),
    ]
    child = subprocess.Popen(
        command,
        cwd=script.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        try:
            peak_rss = max(peak_rss, process.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.001)

    stdout, stderr = child.communicate()
    if child.returncode:
        raise RuntimeError(stderr.strip() or f"worker exited {child.returncode}")
    return json.loads(stdout), peak_rss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=[case.name for case in CASES])
    parser.add_argument(
        "--implementation",
        choices=["reference", "batched"],
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if args.worker:
        if args.case is None or args.implementation is None:
            parser.error("worker mode requires --case and --implementation")
        case = next(case for case in CASES if case.name == args.case)
        _worker(case, args.implementation, args.repeats)
        return

    script = Path(__file__).resolve()
    print(
        "case              reference(s)  batched(s)  speedup  "
        "peak RSS ref/batched (MiB)"
    )
    for case in CASES:
        reference, reference_rss = _measure_in_subprocess(
            script,
            case,
            "reference",
            args.repeats,
        )
        batched, batched_rss = _measure_in_subprocess(
            script,
            case,
            "batched",
            args.repeats,
        )
        speedup = reference["median_seconds"] / batched["median_seconds"]
        print(
            f"{case.name:<17} "
            f"{reference['median_seconds']:>11.4f}  "
            f"{batched['median_seconds']:>10.4f}  "
            f"{speedup:>7.2f}x  "
            f"{reference_rss / 1024**2:>8.1f}/"
            f"{batched_rss / 1024**2:.1f}"
        )


if __name__ == "__main__":
    main()
