"""Explicit timestamp/frame mapping; blank lines retain their original frame slot."""
from pathlib import Path


def timestamp_file_map(sequence_root, timestamp_file, domain, suffix):
    root = Path(sequence_root)
    path = root / timestamp_file
    lines = path.read_text().splitlines()
    files = sorted((root / domain).glob("*" + suffix))
    if len(files) != len(lines):
        raise ValueError(f"Timestamp/file length mismatch: {path}: {len(lines)} vs {len(files)}")
    master_path = root / "timestamps.txt"
    master = master_path.read_text().splitlines() if master_path.exists() else None
    if master is not None and len(master) != len(lines):
        raise ValueError(f"Filtered/full timestamp length mismatch: {path}")
    mapping = {}
    for index, (line, file) in enumerate(zip(lines, files)):
        line = line.strip()
        if not line:
            continue
        timestamp = int(line)
        if timestamp in mapping:
            raise ValueError(f"Duplicate timestamp in {path}: {timestamp}")
        if master is not None and int(master[index]) != timestamp:
            raise ValueError(f"Filtered timestamp shifted frame slot: {path}:{index+1}")
        mapping[timestamp] = str(file)
    return mapping
