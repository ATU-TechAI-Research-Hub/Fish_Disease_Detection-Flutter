from __future__ import annotations

import base64
import gzip
import json

from train.shrimp.prepare_public_dataset import _normalise_labels, load_metadata


def test_load_metadata_accepts_cdp_base64_gzip_wrapper(tmp_path):
    rows = [
        {
            "id": "source-1",
            "owner": "owner-1",
            "split": "train",
            "annotations": {"Shrimp": "4 0.1 0.1 0.2 0.1 0.2 0.2\n"},
        }
    ]
    encoded = base64.b64encode(gzip.compress(json.dumps(rows).encode())).decode()
    wrapper = {"result": {"result": {"type": "string", "value": encoded}}}
    metadata = tmp_path / "cdp.json"
    metadata.write_text(json.dumps(wrapper), encoding="utf-8")

    assert load_metadata(metadata) == rows


def test_normalise_labels_merges_source_classes_to_shrimp():
    labels, count = _normalise_labels(
        {
            "Shrimp": (
                "4 0.1 0.1 0.2 0.1 0.2 0.2\n"
                "2 0.3 0.3 0.4 0.3 0.4 0.4\n"
            )
        }
    )

    assert count == 2
    assert labels.splitlines() == [
        "0 0.1 0.1 0.2 0.1 0.2 0.2",
        "0 0.3 0.3 0.4 0.3 0.4 0.4",
    ]


def test_normalise_labels_skips_malformed_polygons():
    labels, count = _normalise_labels(
        {
            "Shrimp": (
                "0 0.1 0.1 0.2 0.2\n"
                "0 0.1 0.1 2.0 0.2 0.3 0.3\n"
                "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
            )
        }
    )

    assert count == 1
    assert labels == "0 0.1 0.1 0.2 0.1 0.2 0.2\n"
