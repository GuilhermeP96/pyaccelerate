"""
pyaccelerate.cli — Command-line interface.

Provides quick access to hardware detection, benchmarks and diagnostics::

    pyaccelerate info          # Show full engine report
    pyaccelerate benchmark     # Run micro-benchmarks
    pyaccelerate gpu           # GPU detection details
    pyaccelerate status        # One-line status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyaccelerate",
        description="PyAccelerate — High-performance Python acceleration engine",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Full engine report")

    # status
    sub.add_parser("status", help="One-line status")

    # benchmark
    bench_p = sub.add_parser("benchmark", help="Run micro-benchmarks")
    bench_p.add_argument("--full", action="store_true", help="Run full (slower) suite")
    bench_p.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # gpu
    sub.add_parser("gpu", help="GPU detection details")

    # cpu
    sub.add_parser("cpu", help="CPU detection details")

    # virt
    sub.add_parser("virt", help="Virtualization detection")

    # npu
    sub.add_parser("npu", help="NPU detection details")

    # android
    sub.add_parser("android", help="Android/ARM device details")

    # memory
    sub.add_parser("memory", help="Memory stats")

    # priority
    prio_p = sub.add_parser("priority", help="Task priority & energy management")
    prio_p.add_argument("--set", choices=["idle", "below-normal", "normal", "above-normal", "high", "realtime"],
                        help="Set task priority")
    prio_p.add_argument("--energy", choices=["power-saver", "balanced", "performance", "ultra-performance"],
                        help="Set energy profile")
    prio_p.add_argument("--preset", choices=["max", "balanced", "saver"],
                        help="Apply a preset (max/balanced/saver)")

    # max-mode
    max_p = sub.add_parser("max-mode", help="Show max-mode hardware manifest")

    # version
    sub.add_parser("version", help="Show version")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "version":
        from pyaccelerate import __version__
        print(f"pyaccelerate {__version__}")
        return

    if args.command == "info":
        from pyaccelerate.engine import Engine
        engine = Engine()
        print(engine.summary())
        return

    if args.command == "status":
        from pyaccelerate.engine import Engine
        engine = Engine()
        print(engine.status_line())
        return

    if args.command == "benchmark":
        from pyaccelerate.benchmark import run_all
        print("Running benchmarks...")
        results = run_all(quick=not args.full)
        if args.as_json:
            print(json.dumps(results, indent=2))
        else:
            for name, data in results.items():
                print(f"\n{'─' * 50}")
                print(f"  {name}")
                print(f"{'─' * 50}")
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f"  {k}: {v}")
        return

    if args.command == "gpu":
        from pyaccelerate.gpu import detect_all, get_install_hint
        gpus = detect_all()
        if not gpus:
            print("No GPU detected.")
        for i, g in enumerate(gpus):
            print(f"\n[{i}] {g.short_label()}")
            print(f"    Vendor: {g.vendor}  |  Backend: {g.backend}")
            print(f"    VRAM: {g.memory_gb:.1f} GB  |  CUs: {g.compute_units}")
            print(f"    Discrete: {g.is_discrete}  |  Score: {g.score}")
            print(f"    Usable: {g.usable}")
        hint = get_install_hint()
        if hint:
            print(f"\n{hint}")
        return

    if args.command == "cpu":
        from pyaccelerate.cpu import detect
        info = detect()
        print(f"Brand:          {info.brand}")
        print(f"Architecture:   {info.arch}")
        print(f"Physical cores: {info.physical_cores}")
        print(f"Logical cores:  {info.logical_cores}")
        print(f"Frequency:      {info.frequency_mhz:.0f} MHz (boost: {info.frequency_max_mhz:.0f} MHz)")
        print(f"NUMA nodes:     {info.numa_nodes}")
        print(f"SMT ratio:      {info.smt_ratio:.1f}x")
        if info.is_arm:
            print(f"ARM:            yes")
            if info.soc_name:
                print(f"SoC:            {info.soc_name}")
            print(f"NEON:           {info.has_neon}")
            print(f"SVE:            {info.has_sve}")
            if info.arm_clusters:
                for name, cpus in info.arm_clusters.items():
                    print(f"  Cluster:      {name} × {len(cpus)} (CPUs: {cpus})")
            if info.is_android:
                print(f"Android:        yes")
        if info.flags:
            print(f"ISA flags:      {', '.join(info.flags)}")
        return

    if args.command == "android":
        from pyaccelerate.android import (
            is_android, is_termux, is_arm, get_device_info, get_soc_info,
            get_thermal_zones, get_battery_info, is_thermally_throttled,
            detect_big_little, get_arm_features, get_cpu_freq_info,
        )
        if not is_arm():
            print("Not running on ARM architecture.")
            return
        print(f"ARM:       yes ({__import__('platform').machine()})")
        print(f"Android:   {is_android()}")
        print(f"Termux:    {is_termux()}")

        # Device info
        dev = get_device_info()
        if dev:
            print("\n── Device Info ──")
            for k, v in dev.items():
                print(f"  {k}: {v}")

        # SoC
        soc = get_soc_info()
        if soc:
            print(f"\n── SoC: {soc.name} ({soc.vendor}) ──")
            print(f"  CPU arch:  {soc.cpu_arch}")
            cores = f"{soc.cpu_cores_big}B"
            if soc.cpu_cores_mid:
                cores += f"+{soc.cpu_cores_mid}M"
            cores += f"+{soc.cpu_cores_little}L"
            print(f"  CPU cores: {cores}")
            print(f"  GPU:       {soc.gpu_name} ({soc.gpu_cores} cores)")
            print(f"  NPU:       {soc.npu_name} ({soc.npu_tops:.1f} TOPS)")
            print(f"  Process:   {soc.process_nm} nm")

        # big.LITTLE
        bl = detect_big_little()
        if bl:
            print("\n── big.LITTLE Clusters ──")
            for name, cpus in bl.items():
                print(f"  {name}: CPUs {cpus}")

        # ARM features
        feats = get_arm_features()
        if feats:
            print(f"\n── ARM Features ──")
            print(f"  {', '.join(feats)}")

        # Frequencies
        freqs = get_cpu_freq_info()
        if freqs:
            print("\n── CPU Frequencies ──")
            for cpu_name, fmap in freqs.items():
                cur = fmap.get('scaling_cur_freq', 0)
                mx = fmap.get('cpuinfo_max_freq', fmap.get('scaling_max_freq', 0))
                print(f"  {cpu_name}: {cur:.0f} MHz (max {mx:.0f} MHz)")

        # Thermal
        temps = get_thermal_zones()
        if temps:
            print(f"\n── Thermal Zones ──")
            for name, temp in temps.items():
                print(f"  {name}: {temp:.1f}°C")
            if is_thermally_throttled():
                print("  ⚠  THERMALLY THROTTLED")

        # Battery
        batt = get_battery_info()
        if batt:
            print(f"\n── Battery ──")
            for k, v in batt.items():
                print(f"  {k}: {v}")
        return

    if args.command == "virt":
        from pyaccelerate.virt import detect
        vi = detect()
        parts = vi.summary_parts()
        if parts:
            print("Detected:", ", ".join(parts))
        else:
            print("No virtualization features detected.")
        return

    if args.command == "npu":
        from pyaccelerate.npu import detect_all as detect_npus, get_install_hint as npu_hint
        npus = detect_npus()
        if not npus:
            print("No NPU detected.")
        for i, n in enumerate(npus):
            print(f"\n[{i}] {n.short_label()}")
            print(f"    Vendor: {n.vendor}  |  Backend: {n.backend}")
            print(f"    TOPS: {n.tops:.1f}  |  Score: {n.score}")
            print(f"    Driver: {n.driver_version or 'N/A'}")
            print(f"    Usable: {n.usable}")
        hint = npu_hint()
        if hint:
            print(f"\n{hint}")
        return

    if args.command == "memory":
        from pyaccelerate.memory import get_stats, get_pressure
        stats = get_stats()
        pressure = get_pressure()
        print(f"Pressure: {pressure.name}")
        for k, v in stats.items():
            if k == "error":
                continue
            print(f"  {k}: {v:.2f}")
        return

    if args.command == "priority":
        from pyaccelerate.priority import (
            TaskPriority as _TP, EnergyProfile as _EP,
            set_task_priority as _stp, set_energy_profile as _sep,
            get_priority_info as _gpi,
            max_performance as _mp, balanced as _bal, power_saver as _ps,
        )

        if args.preset:
            presets = {"max": _mp, "balanced": _bal, "saver": _ps}
            result = presets[args.preset]()
            print(f"Preset '{args.preset}' applied: {result}")
            return

        if args.set:
            p_map = {
                "idle": _TP.IDLE, "below-normal": _TP.BELOW_NORMAL,
                "normal": _TP.NORMAL, "above-normal": _TP.ABOVE_NORMAL,
                "high": _TP.HIGH, "realtime": _TP.REALTIME,
            }
            ok = _stp(p_map[args.set])
            print(f"Task priority → {args.set}: {'OK' if ok else 'FAILED'}")

        if args.energy:
            e_map = {
                "power-saver": _EP.POWER_SAVER, "balanced": _EP.BALANCED,
                "performance": _EP.PERFORMANCE,
                "ultra-performance": _EP.ULTRA_PERFORMANCE,
            }
            ok = _sep(e_map[args.energy])
            print(f"Energy profile → {args.energy}: {'OK' if ok else 'FAILED'}")

        if not args.set and not args.energy:
            info = _gpi()
            print("Current priority settings:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        return

    if args.command == "max-mode":
        from pyaccelerate.max_mode import MaxMode
        with MaxMode(set_priority=False, set_energy=False) as m:
            print(m.summary())
        return


if __name__ == "__main__":
    main()
