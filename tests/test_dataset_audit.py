from pathlib import Path

from training.audit_dataset import audit


def test_audit_accepts_annotated_person_and_empty_negative(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "cam1__000001.jpg").write_bytes(b"x")
    (images / "cam1__000002.jpg").write_bytes(b"x")
    (labels / "cam1__000001.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    (labels / "cam1__000002.txt").write_text("")
    result = audit(images, labels)
    assert result["valid"] and result["persons"] == 1
    assert result["verified_negative_images"] == 1


def test_audit_rejects_missing_and_out_of_bounds_labels(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "cam1__000001.jpg").write_bytes(b"x")
    (images / "cam1__000002.jpg").write_bytes(b"x")
    (labels / "cam1__000001.txt").write_text("0 0.9 0.9 0.5 0.5\n")
    result = audit(images, labels)
    assert not result["valid"]
    assert result["missing_labels"] == 1
    assert result["error_count"] == 2


def test_audit_rejects_empty_collection(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    try:
        audit(images, labels)
    except ValueError as exc:
        assert "No supported images" in str(exc)
    else:
        raise AssertionError("Empty dataset must be rejected")
