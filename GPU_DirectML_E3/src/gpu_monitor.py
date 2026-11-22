from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


POWERSHELL_SAMPLE = r"""
$ErrorActionPreference = 'Stop'
$targetPid = 0
try {
    $engineRows = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction Stop |
        Where-Object { $_.Name -match "(^|_)pid_${targetPid}(_|$)" })
    $processRows = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop |
        Where-Object { $_.Name -match "(^|_)pid_${targetPid}(_|$)" })
    $engineMax = $null
    if ($engineRows.Count -gt 0) {
        $engineMax = [double](($engineRows | Measure-Object -Property UtilizationPercentage -Maximum).Maximum)
    }
    $dedicated = 0.0
    $shared = 0.0
    $luids = @()
    foreach ($row in $processRows) {
        if ($null -ne $row.DedicatedUsage) { $dedicated += [double]$row.DedicatedUsage }
        if ($null -ne $row.SharedUsage) { $shared += [double]$row.SharedUsage }
        $match = [regex]::Match($row.Name, 'luid_0x[0-9a-fA-F]+_0x[0-9a-fA-F]+_phys_[0-9]+')
        if ($match.Success -and $luids -notcontains $match.Value) { $luids += $match.Value }
    }
    $adapterRows = @(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -ErrorAction Stop)
    $adapterDedicated = 0.0
    $adapterShared = 0.0
    foreach ($adapter in $adapterRows) {
        $belongs = $false
        foreach ($luid in $luids) { if ($adapter.Name -like "*$luid*") { $belongs = $true } }
        if ($belongs) {
            if ($null -ne $adapter.DedicatedUsage) { $adapterDedicated += [double]$adapter.DedicatedUsage }
            if ($null -ne $adapter.SharedUsage) { $adapterShared += [double]$adapter.SharedUsage }
        }
    }
    [pscustomobject]@{
        ok = $true
        gpu_engine_max_percent = $engineMax
        gpu_engine_instances = $engineRows.Count
        process_dedicated_bytes = $dedicated
        process_shared_bytes = $shared
        adapter_dedicated_bytes = $adapterDedicated
        adapter_shared_bytes = $adapterShared
        adapter_luids = ($luids -join ';')
        error = $null
    } | ConvertTo-Json -Compress
} catch {
    [pscustomobject]@{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress
}
"""


class GPUUsageMonitor:
    """Amostra uso da GPU DirectX/Windows, memória, RAM e CPU durante o treinamento."""
    FIELDS = [
        "timestamp", "elapsed_seconds", "python_pid", "gpu_engine_max_percent",
        "gpu_engine_instances", "gpu_process_dedicated_mb", "gpu_process_shared_mb",
        "gpu_adapter_dedicated_mb", "gpu_adapter_shared_mb", "gpu_adapter_luids",
        "process_ram_mb", "system_ram_used_percent", "process_cpu_percent", "counter_status",
    ]

    def __init__(self, output_dir: Path, interval_seconds: float = 2.0) -> None:
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self.csv_path = output_dir / "gpu_monitoramento.csv"
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = 0.0
        self.process = psutil.Process(os.getpid())
        self.rows: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def _query_windows_gpu(self) -> dict[str, Any]:
        if os.name != "nt":
            return {"ok": False, "error": "Monitoramento de GPU configurado para Windows."}
        command = [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-Command", POWERSHELL_SAMPLE.replace("$targetPid = 0", f"$targetPid = {os.getpid()}"),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=12, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"PowerShell retornou {completed.returncode}")
            data = json.loads(completed.stdout.strip())
            return data
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _sample_once(self) -> dict[str, Any]:
        gpu = self._query_windows_gpu()
        try:
            ram_mb = self.process.memory_info().rss / (1024 ** 2)
        except Exception:
            ram_mb = None
        try:
            system_ram = psutil.virtual_memory().percent
        except Exception:
            system_ram = None
        try:
            cpu_pct = self.process.cpu_percent(interval=None)
        except Exception:
            cpu_pct = None

        to_mb = lambda value: (float(value) / (1024 ** 2)) if value is not None else None
        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
            "python_pid": os.getpid(),
            "gpu_engine_max_percent": gpu.get("gpu_engine_max_percent") if gpu.get("ok") else None,
            "gpu_engine_instances": gpu.get("gpu_engine_instances") if gpu.get("ok") else None,
            "gpu_process_dedicated_mb": to_mb(gpu.get("process_dedicated_bytes")) if gpu.get("ok") else None,
            "gpu_process_shared_mb": to_mb(gpu.get("process_shared_bytes")) if gpu.get("ok") else None,
            "gpu_adapter_dedicated_mb": to_mb(gpu.get("adapter_dedicated_bytes")) if gpu.get("ok") else None,
            "gpu_adapter_shared_mb": to_mb(gpu.get("adapter_shared_bytes")) if gpu.get("ok") else None,
            "gpu_adapter_luids": gpu.get("adapter_luids") if gpu.get("ok") else None,
            "process_ram_mb": round(ram_mb, 2) if ram_mb is not None else None,
            "system_ram_used_percent": round(system_ram, 2) if system_ram is not None else None,
            "process_cpu_percent": round(cpu_pct, 2) if cpu_pct is not None else None,
            "counter_status": "OK" if gpu.get("ok") else f"INDISPONÍVEL: {gpu.get('error', 'sem detalhe')}",
        }
        return row

    def _run(self) -> None:
        self.process.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            row = self._sample_once()
            self.rows.append(row)
            with self.csv_path.open("a", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            if row["counter_status"] != "OK":
                self.last_error = row["counter_status"]
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name="GPUUsageMonitor", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
        good = [row for row in self.rows if row["counter_status"] == "OK"]

        def stats(field: str) -> dict[str, float | None]:
            values = [float(row[field]) for row in good if row.get(field) is not None]
            if not values:
                return {"average": None, "maximum": None, "samples": 0}
            return {
                "average": round(sum(values) / len(values), 3),
                "maximum": round(max(values), 3),
                "samples": len(values),
            }

        summary = {
            "monitoring_interval_seconds": self.interval_seconds,
            "total_samples": len(self.rows),
            "successful_gpu_samples": len(good),
            "gpu_engine_max_percent": stats("gpu_engine_max_percent"),
            "gpu_process_dedicated_mb": stats("gpu_process_dedicated_mb"),
            "gpu_process_shared_mb": stats("gpu_process_shared_mb"),
            "gpu_adapter_dedicated_mb": stats("gpu_adapter_dedicated_mb"),
            "gpu_adapter_shared_mb": stats("gpu_adapter_shared_mb"),
            "process_ram_mb": stats("process_ram_mb"),
            "system_ram_used_percent": stats("system_ram_used_percent"),
            "process_cpu_percent": stats("process_cpu_percent"),
            "last_counter_error": self.last_error,
            "notes": (
                "Os contadores do Windows são amostras por processo/adaptador. A utilização registrada é o maior "
                "percentual entre os motores GPU associados ao processo Python, não uma média de toda a placa. "
                "Memória dedicada/compartilhada do processo é separada do total do adaptador."
            ),
        }
        (self.output_dir / "gpu_monitoramento_resumo.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
