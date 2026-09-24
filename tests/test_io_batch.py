import numpy as np
from PIL import Image

from picme.batch import collect_images, output_path, run_batch
from picme.cli import main
from picme.io import load_image, save_image
from picme.settings import get_preset


def _write(path, arr):
    Image.fromarray(arr[..., ::-1]).save(path)


def test_exif_orientation_applied(tmp_path):
    arr = np.zeros((20, 40, 3), np.uint8)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6  # 90° 回転
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif.tobytes())
    img, meta = load_image(p)
    assert img.shape[:2] == (40, 20)
    out = tmp_path / "out.jpg"
    save_image(out, img, meta)
    with Image.open(out) as re:
        assert re.getexif().get(0x0112) == 1
        assert re.size == (20, 40)


def test_batch_and_cli(tmp_path, portrait_like):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    _write(src / "a.jpg", portrait_like)
    _write(src / "sub" / "b.png", portrait_like)
    (src / "note.txt").write_text("x")

    assert [p.name for p in collect_images([src])] == ["a.jpg"]
    files = collect_images([src], recursive=True)
    assert {p.name for p in files} == {"a.jpg", "b.png"}

    out = tmp_path / "out"
    results = run_batch(files, out, get_preset("natural"), workers=1, use_ai=False, suffix="_r")
    assert all(r.error is None for r in results)
    assert (out / "a_r.jpg").exists() and (out / "b_r.png").exists()

    assert main([str(src), "-o", str(tmp_path / "cli"), "--no-ai", "-j", "1", "--format", "png"]) == 0
    assert (tmp_path / "cli" / "a.png").exists()


def test_output_path():
    from pathlib import Path

    assert output_path(Path("x/a.JPG"), Path("o"), "_s", "png") == Path("o/a_s.png")
