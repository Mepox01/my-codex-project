from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result_output import generate_report


def test_generate_report_creates_summary_and_photo(tmp_path: Path) -> None:
    photo_path = tmp_path / "sample_photo.png"

    summary, generated_path = generate_report(str(photo_path))

    assert "Result Summary" in summary
    assert "Accuracy: 95%" in summary
    assert generated_path == photo_path
    assert generated_path.exists()
    assert generated_path.read_bytes().startswith(b"\x89PNG")
