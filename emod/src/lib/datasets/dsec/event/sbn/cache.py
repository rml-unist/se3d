"""Read the transferred, trusted SE3D sparse event stacks without opening HDF5.

The .npy suffix is historical: these files are torch.save pickles containing
NumPy arrays, not NumPy .npy files. Cache-only mode never repairs a file during
training; preparation must finish before a frozen split is consumed.
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_trusted_cache(path, stack_size, height, width, num_of_future_event=0, inspection=None):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    try:
        data = torch.load(path, **kwargs)
        if not isinstance(data, dict) or set(data) != {"left", "right"}:
            raise ValueError("expected left/right stereo dictionary")
        for side in ("left", "right"):
            stacks = data[side]
            allowed_counts = (1, 2) if num_of_future_event == 0 else (2,)
            if not isinstance(stacks, list) or len(stacks) not in allowed_counts:
                raise ValueError(f"{side}: unexpected past/future stack count")
            for sparse in stacks:
                for key in ("index", "stacked_polarity"):
                    if key not in sparse or len(sparse[key]) != stack_size:
                        raise ValueError(f"{side}: invalid {key} stack count")
                for idx, polarity in zip(sparse["index"], sparse["stacked_polarity"]):
                    idx, polarity = np.asarray(idx), np.asarray(polarity)
                    if idx.ndim != 1 or polarity.shape != idx.shape:
                        raise ValueError(f"{side}: inconsistent sparse arrays")
                    if not np.issubdtype(idx.dtype, np.integer):
                        raise ValueError(f"{side}: noninteger pixel index")
                    if idx.size and (idx.min() < 0 or idx.max() >= height * width):
                        raise ValueError(f"{side}: pixel index outside image")
                    if not np.isin(polarity, (-1, 1)).all():
                        raise ValueError(f"{side}: invalid sparse polarity")
        if inspection is not None:
            inspection['stored_branches'] = {s: len(data[s]) for s in ('left', 'right')}
        # Some transferred *_10_0 caches contain the legacy extra branch.
        # Original EMOD explicitly slices t[:1] before the network. Enforce
        # that same past-only input here when no future events are requested,
        # after validating every stored branch; do not modify or drop a frame.
        if num_of_future_event == 0:
            data = {side: branches[:1] for side, branches in data.items()}
        return data
    except Exception as exc:
        raise RuntimeError(f"Missing or invalid SE3D cache: {path}; prepare it before training") from exc


def load_time_bounds(path, event_root, num_of_event, stack_method, stack_size,
                     num_of_future_event):
    if not path:
        raise ValueError("cache_only requires an explicit cache_time_bounds metadata path")
    with open(path) as stream:
        metadata = json.load(stream)
    for key, value in (("num_of_event", num_of_event), ("stack_method", stack_method),
                       ("stack_size", stack_size), ("num_of_future_event", num_of_future_event)):
        if metadata.get(key) != value:
            raise ValueError(f"Cache time bounds {key} mismatch: {metadata.get(key)!r} != {value!r}")
    root = Path(event_root)
    sequence = "/".join(root.parent.parts[-2:])
    if root.name != "events" or sequence not in metadata["sequences"]:
        raise ValueError(f"No SE3D time bounds for {event_root}")
    bounds = metadata["sequences"][sequence]
    sides = {}
    for side in ("left", "right"):
        item = bounds["sides"][side]
        sides[side] = SimpleNamespace(
            t_offset=int(item["t_offset_used_by_original_loader"]),
            min_time=int(item["min_time"]), max_time=int(item["max_time"]),
            t_final=int(item["t_final"]), total_event=int(item["total_events"]),
        )
        if sides[side].min_time > sides[side].max_time:
            raise ValueError(f"Invalid time bounds for {sequence}/{side}")
    return sides
