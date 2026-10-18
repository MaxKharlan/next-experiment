import pytest

from bench.calibration import reliability
from bench.reliability_plot import plot_report


def test_plot_preserves_observed_coordinates_and_empty_bins(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    metrics = reliability([(0.1, False), (0.8, True), (1.0, False)], bins=5)
    report = {"split": "dev", "methods": {"example": {"metrics": {"all": metrics}}}}
    closed = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: closed.append(fig))
    plot_report(report, tmp_path)
    fig = closed[-1]
    try:
        points = fig.axes[0].collections[0].get_offsets().tolist()
        assert points == [[0.1, 0.0], [0.9, 0.5]]
        assert [bar.get_height() for bar in fig.axes[1].patches] == [1, 0, 0, 0, 2]
        assert (tmp_path / "reliability.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (tmp_path / "reliability.svg").read_text()
    finally:
        close(fig)
