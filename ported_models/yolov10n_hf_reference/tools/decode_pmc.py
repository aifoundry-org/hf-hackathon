#!/usr/bin/env python3
"""Decode a versioned YR PMC region embedded in a full launcher dump.

The byte offset is intentionally mandatory: launcher dumps can contain several
unrelated regions, and silently scanning for a magic value could accept stale
data from an earlier run.
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REGION_MAGIC = 0x4D505259       # little-endian bytes b"YRPM"
HART_MAGIC = 0x48505259         # little-endian bytes b"YRPH"
AGGREGATE_MAGIC = 0x41505259    # little-endian bytes b"YRPA"
FORMAT_VERSION = 1
ENDIAN_MARKER = 0x01020304

HPM_COUNT = 6
SC_BANKS = 4
MS_COUNT = 8
COUNTERS_PER_BLOCK = 3
MAX_HARTS = 32

HEADER_BYTES = 128
HART_RECORD_BYTES = 128
AGGREGATE_OFFSET = HEADER_BYTES + MAX_HARTS * HART_RECORD_BYTES
AGGREGATE_BYTES = 592
REGION_BYTES = AGGREGATE_OFFSET + AGGREGATE_BYTES

FLAG_SC_REQUESTED = 1 << 0
FLAG_MS_REQUESTED = 1 << 1
KNOWN_FLAGS = FLAG_SC_REQUESTED | FLAG_MS_REQUESTED
ERROR_VALUE = (1 << 64) - 1

HEADER_STRUCT = struct.Struct("<8I96x")
HART_STRUCT = struct.Struct("<4I14Q")
AGGREGATE_STRUCT = struct.Struct("<4I72Q")

HPM_EVENTS: Sequence[Tuple[str, str]] = (
    ("hpmcounter3", "minion_cycles"),
    ("hpmcounter4", "retired_instructions_thread_0"),
    ("hpmcounter5", "retired_instructions_thread_1"),
    ("hpmcounter6", "l2_miss_requests"),
    ("hpmcounter7", "minion_icache_requests"),
    ("hpmcounter8", "icache_etlink_requests"),
)
SC_EVENTS: Sequence[str] = ("cycles", "all_l2_reads", "all_l2_writes")
MS_EVENTS: Sequence[str] = ("cycles", "all_mesh_reads", "all_mesh_writes")


def parse_integer(value: str) -> int:
    """Parse decimal or a Python-style base-prefixed integer such as 0x10000."""
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected an integer byte offset (decimal or 0x-prefixed)"
        ) from exc


def hex32(value: int) -> str:
    return "0x{:08x}".format(value)


def hex64(value: int) -> str:
    return "0x{:016x}".format(value)


def counter_delta(start: int, end: int) -> Tuple[int, bool]:
    """Return the unsigned 64-bit delta and whether the counter wrapped."""
    return ((end - start) & ERROR_VALUE, end < start)


def magic_report(actual: int, expected: int, expected_ascii: str) -> Dict[str, Any]:
    return {
        "actual": actual,
        "actual_hex": hex32(actual),
        "expected": expected,
        "expected_hex": hex32(expected),
        "expected_ascii": expected_ascii,
        "valid": actual == expected,
    }


def new_report(path: Path, dump_bytes: int, offset: int) -> Dict[str, Any]:
    return {
        "status": "FAIL",
        "dump": str(path),
        "dump_bytes": dump_bytes,
        "offset": offset,
        "format": {
            "name": "YR PMC",
            "byte_order": "little",
            "expected_magic_ascii": "YRPM",
            "expected_magic_hex": hex32(REGION_MAGIC),
            "expected_version": FORMAT_VERSION,
            "expected_region_bytes": REGION_BYTES,
        },
        "header": None,
        "harts": [],
        "shared": None,
        "errors": [],
        "warnings": [],
    }


def add_error(report: Dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def add_warning(report: Dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def decode_hart(
    blob: bytes, region_offset: int, slot: int, report: Dict[str, Any]
) -> Dict[str, Any]:
    record_offset = region_offset + HEADER_BYTES + slot * HART_RECORD_BYTES
    values = HART_STRUCT.unpack_from(blob, record_offset)
    magic, hart_id, minion_id, thread_id = values[:4]
    starts = values[4:4 + HPM_COUNT]
    ends = values[4 + HPM_COUNT:4 + 2 * HPM_COUNT]
    errors: List[str] = []

    if magic != HART_MAGIC:
        errors.append(
            "hart slot {} magic {} != {}".format(
                slot, hex32(magic), hex32(HART_MAGIC)
            )
        )
    if hart_id != slot:
        errors.append(
            "hart slot {} records hart_id {}".format(slot, hart_id)
        )

    counters: List[Dict[str, Any]] = []
    for index, ((csr_name, event_name), start, end) in enumerate(
        zip(HPM_EVENTS, starts, ends)
    ):
        delta, wrapped = counter_delta(start, end)
        counters.append(
            {
                "index": index,
                "csr": csr_name,
                "event": event_name,
                "start": start,
                "start_hex": hex64(start),
                "end": end,
                "end_hex": hex64(end),
                "delta": delta,
                "delta_hex": hex64(delta),
                "wrapped": wrapped,
                "status": "PASS",
            }
        )

    for message in errors:
        add_error(report, message)
    return {
        "status": "PASS" if not errors else "FAIL",
        "slot": slot,
        "record_offset": record_offset,
        "magic": magic_report(magic, HART_MAGIC, "YRPH"),
        "hart_id": hart_id,
        "minion_id": minion_id,
        "thread_id": thread_id,
        "counters": counters,
        "errors": errors,
    }


def decode_shared_family(
    family: str,
    requested: bool,
    supported_mask: int,
    block_count: int,
    event_names: Sequence[str],
    starts: Sequence[int],
    ends: Sequence[int],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    entry_count = block_count * COUNTERS_PER_BLOCK
    valid_mask = (1 << entry_count) - 1
    unknown_mask = supported_mask & ~valid_mask
    family_upper = family.upper()
    samples: List[Dict[str, Any]] = []

    if unknown_mask:
        add_error(
            report,
            "{} supported mask has out-of-range bits: {}".format(
                family_upper, hex32(unknown_mask)
            ),
        )
    if not requested and supported_mask:
        add_error(
            report,
            "{} reports supported samples although it was not requested".format(
                family_upper
            ),
        )
    if requested and not (supported_mask & valid_mask):
        add_warning(
            report,
            "{} sampling was requested but this firmware/emulator reported "
            "no supported counters".format(family_upper),
        )

    for block in range(block_count):
        for counter in range(COUNTERS_PER_BLOCK):
            flat_index = block * COUNTERS_PER_BLOCK + counter
            bit = 1 << flat_index
            start = starts[flat_index]
            end = ends[flat_index]
            supported = bool(supported_mask & bit)
            start_valid = start != ERROR_VALUE
            end_valid = end != ERROR_VALUE
            status = "NOT_REQUESTED"
            delta: Optional[int] = None
            wrapped: Optional[bool] = None

            if requested:
                status = "PASS" if supported else "UNSUPPORTED"
            if supported:
                if not start_valid or not end_valid:
                    status = "FAIL"
                    add_error(
                        report,
                        "{} block {} counter {} is marked supported but has "
                        "an error sentinel endpoint".format(
                            family_upper, block, counter
                        ),
                    )
                else:
                    delta, wrapped = counter_delta(start, end)
            elif requested and start_valid and end_valid:
                status = "FAIL"
                add_error(
                    report,
                    "{} block {} counter {} has two valid endpoints but its "
                    "support bit is clear".format(
                        family_upper, block, counter
                    ),
                )
            elif requested and start_valid != end_valid:
                status = "FAIL"
                add_error(
                    report,
                    "{} block {} counter {} was available at only one "
                    "boundary".format(family_upper, block, counter),
                )

            sample = {
                "status": status,
                "block": block,
                "block_kind": "bank" if family == "sc" else "memory_shire",
                "counter": counter,
                "event": event_names[counter],
                "supported": supported,
                "start": start if start_valid else None,
                "start_hex": hex64(start) if start_valid else None,
                "end": end if end_valid else None,
                "end_hex": hex64(end) if end_valid else None,
                "delta": delta,
                "delta_hex": hex64(delta) if delta is not None else None,
                "wrapped": wrapped,
            }
            samples.append(sample)

    return {
        "requested": requested,
        "supported": bool(supported_mask & valid_mask),
        "supported_count": bin(supported_mask & valid_mask).count("1"),
        "entry_count": entry_count,
        "supported_mask": supported_mask,
        "supported_mask_hex": hex32(supported_mask),
        "samples": samples,
    }


def decode_aggregate(
    blob: bytes, region_offset: int, flags: int, report: Dict[str, Any]
) -> Dict[str, Any]:
    error_count_before = len(report["errors"])
    record_offset = region_offset + AGGREGATE_OFFSET
    values = AGGREGATE_STRUCT.unpack_from(blob, record_offset)
    magic, shire_id, sc_supported_mask, ms_supported_mask = values[:4]
    cursor = 4
    sc_entries = SC_BANKS * COUNTERS_PER_BLOCK
    ms_entries = MS_COUNT * COUNTERS_PER_BLOCK
    sc_start = values[cursor:cursor + sc_entries]
    cursor += sc_entries
    sc_end = values[cursor:cursor + sc_entries]
    cursor += sc_entries
    ms_start = values[cursor:cursor + ms_entries]
    cursor += ms_entries
    ms_end = values[cursor:cursor + ms_entries]

    if magic != AGGREGATE_MAGIC:
        add_error(
            report,
            "aggregate magic {} != {}".format(
                hex32(magic), hex32(AGGREGATE_MAGIC)
            ),
        )

    sc = decode_shared_family(
        "sc",
        bool(flags & FLAG_SC_REQUESTED),
        sc_supported_mask,
        SC_BANKS,
        SC_EVENTS,
        sc_start,
        sc_end,
        report,
    )
    ms = decode_shared_family(
        "ms",
        bool(flags & FLAG_MS_REQUESTED),
        ms_supported_mask,
        MS_COUNT,
        MS_EVENTS,
        ms_start,
        ms_end,
        report,
    )
    return {
        "status": (
            "PASS"
            if len(report["errors"]) == error_count_before
            else "FAIL"
        ),
        "record_offset": record_offset,
        "magic": magic_report(magic, AGGREGATE_MAGIC, "YRPA"),
        "shire_id": shire_id,
        "sc": sc,
        "ms": ms,
    }


def decode_blob(blob: bytes, path: Path, offset: int) -> Dict[str, Any]:
    report = new_report(path, len(blob), offset)

    if offset < 0:
        add_error(report, "byte offset must be non-negative")
        return report
    if offset > len(blob):
        add_error(
            report,
            "byte offset {} is beyond the {}-byte dump".format(
                offset, len(blob)
            ),
        )
        return report
    if len(blob) - offset < HEADER_BYTES:
        add_error(
            report,
            "truncated PMC header: need {} bytes at offset {}, have {}".format(
                HEADER_BYTES, offset, len(blob) - offset
            ),
        )
        return report

    (
        magic,
        version,
        region_bytes,
        active_harts,
        hpm_count,
        max_harts,
        flags,
        endian_marker,
    ) = HEADER_STRUCT.unpack_from(blob, offset)
    header = {
        "status": "PASS",
        "record_offset": offset,
        "magic": magic_report(magic, REGION_MAGIC, "YRPM"),
        "version": version,
        "region_bytes": region_bytes,
        "active_harts": active_harts,
        "hpm_count": hpm_count,
        "max_harts": max_harts,
        "flags": {
            "raw": flags,
            "raw_hex": hex32(flags),
            "sc_requested": bool(flags & FLAG_SC_REQUESTED),
            "ms_requested": bool(flags & FLAG_MS_REQUESTED),
            "unknown_bits_hex": hex32(flags & ~KNOWN_FLAGS),
        },
        "endian_marker": endian_marker,
        "endian_marker_hex": hex32(endian_marker),
    }
    report["header"] = header

    header_errors: List[str] = []
    if magic != REGION_MAGIC:
        header_errors.append(
            "region magic {} != {} (expected ASCII YRPM)".format(
                hex32(magic), hex32(REGION_MAGIC)
            )
        )
    if version != FORMAT_VERSION:
        header_errors.append(
            "unsupported format version {}; expected {}".format(
                version, FORMAT_VERSION
            )
        )
    if region_bytes != REGION_BYTES:
        header_errors.append(
            "recorded region size {} != v{} size {}".format(
                region_bytes, FORMAT_VERSION, REGION_BYTES
            )
        )
    if not 1 <= active_harts <= MAX_HARTS:
        header_errors.append(
            "active_harts {} is outside 1..{}".format(
                active_harts, MAX_HARTS
            )
        )
    if hpm_count != HPM_COUNT:
        header_errors.append(
            "hpm_count {} != {}".format(hpm_count, HPM_COUNT)
        )
    if max_harts != MAX_HARTS:
        header_errors.append(
            "max_harts {} != {}".format(max_harts, MAX_HARTS)
        )
    if flags & ~KNOWN_FLAGS:
        header_errors.append(
            "unknown v{} header flag bits {}".format(
                FORMAT_VERSION, hex32(flags & ~KNOWN_FLAGS)
            )
        )
    if endian_marker != ENDIAN_MARKER:
        header_errors.append(
            "endian marker {} != {}".format(
                hex32(endian_marker), hex32(ENDIAN_MARKER)
            )
        )

    for message in header_errors:
        add_error(report, message)
    if header_errors:
        header["status"] = "FAIL"

    available = len(blob) - offset
    if available < REGION_BYTES:
        add_error(
            report,
            "truncated PMC region: need {} bytes at offset {}, have {}".format(
                REGION_BYTES, offset, available
            ),
        )
        return report

    slots_to_decode = min(active_harts, MAX_HARTS)
    report["harts"] = [
        decode_hart(blob, offset, slot, report)
        for slot in range(slots_to_decode)
    ]
    report["shared"] = decode_aggregate(blob, offset, flags, report)
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def render_optional_counter(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("RESULT: {}".format(report["status"]))
    lines.append(
        "dump={} bytes={} offset={}".format(
            report["dump"], report["dump_bytes"], report["offset"]
        )
    )
    lines.append(
        "format={} magic={} version={} region_bytes={}".format(
            report["format"]["name"],
            report["format"]["expected_magic_ascii"],
            report["format"]["expected_version"],
            report["format"]["expected_region_bytes"],
        )
    )

    header = report.get("header")
    if header is not None:
        lines.append(
            "header {} magic={} version={} region_bytes={} active_harts={} "
            "hpm_count={} max_harts={} endian={}".format(
                header["status"],
                header["magic"]["actual_hex"],
                header["version"],
                header["region_bytes"],
                header["active_harts"],
                header["hpm_count"],
                header["max_harts"],
                header["endian_marker_hex"],
            )
        )
        flags = header["flags"]
        lines.append(
            "support_flags raw={} sc_requested={} ms_requested={}".format(
                flags["raw_hex"],
                str(flags["sc_requested"]).lower(),
                str(flags["ms_requested"]).lower(),
            )
        )

    for hart in report.get("harts", []):
        lines.append(
            "hart[{}] {} hart_id={} minion_id={} thread_id={} "
            "record_offset={}".format(
                hart["slot"],
                hart["status"],
                hart["hart_id"],
                hart["minion_id"],
                hart["thread_id"],
                hart["record_offset"],
            )
        )
        for counter in hart["counters"]:
            lines.append(
                "  {} {} start={} end={} delta={} wrapped={} {}".format(
                    counter["csr"],
                    counter["event"],
                    counter["start"],
                    counter["end"],
                    counter["delta"],
                    str(counter["wrapped"]).lower(),
                    counter["status"],
                )
            )

    shared = report.get("shared")
    if shared is not None:
        lines.append(
            "shared {} shire_id={} magic={} record_offset={}".format(
                shared["status"],
                shared["shire_id"],
                shared["magic"]["actual_hex"],
                shared["record_offset"],
            )
        )
        for family_name in ("sc", "ms"):
            family = shared[family_name]
            lines.append(
                "{} requested={} supported={} supported_count={}/{} "
                "mask={}".format(
                    family_name.upper(),
                    str(family["requested"]).lower(),
                    str(family["supported"]).lower(),
                    family["supported_count"],
                    family["entry_count"],
                    family["supported_mask_hex"],
                )
            )
            for sample in family["samples"]:
                lines.append(
                    "  {}[{}].{} start={} end={} delta={} {}".format(
                        sample["block_kind"],
                        sample["block"],
                        sample["event"],
                        render_optional_counter(sample["start"]),
                        render_optional_counter(sample["end"]),
                        render_optional_counter(sample["delta"]),
                        sample["status"],
                    )
                )

    for warning in report.get("warnings", []):
        lines.append("WARNING: {}".format(warning))
    for error in report.get("errors", []):
        lines.append("ERROR: {}".format(error))
    return "\n".join(lines)


def read_and_decode(path: Path, offset: int) -> Dict[str, Any]:
    try:
        blob = path.read_bytes()
    except OSError as exc:
        report = new_report(path, 0, offset)
        add_error(report, "cannot read dump: {}".format(exc))
        return report
    return decode_blob(blob, path, offset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the fixed YR PMC v1 record from a full ET-SoC1 launcher "
            "dump. The exact byte offset is required."
        ),
        epilog=(
            "example: %(prog)s build/slice.dump --offset 0x20000 "
            "--format json"
        ),
    )
    parser.add_argument("dump", type=Path, help="full launcher dump file")
    parser.add_argument(
        "--offset",
        required=True,
        type=parse_integer,
        help="byte offset of struct yr_pmc_region (decimal or 0x-prefixed)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format (default: text)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = read_and_decode(args.dump, args.offset)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(report))
        sys.stdout.write("\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
