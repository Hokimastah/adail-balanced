from adail.benchmarking import extract_adail_metrics, make_comparison_table


def test_comparison_table():
    history = {
        "average_accuracy": 0.75,
        "average_forgetting": 0.1,
        "task_results": [{"metrics": {"seen_mae": 0.3, "seen_qwk": 0.8}}],
    }
    row = extract_adail_metrics(history, "eyepacs", "3-1")
    table = make_comparison_table([row])
    assert "ADAIL" in table
    assert "average_accuracy" in table
    assert "75.0" in table
