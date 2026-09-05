from adail.baselines import build_lo2ln_command, command_to_string, get_lo2ln_experiment


def test_lo2ln_experiment_mapping():
    exp = get_lo2ln_experiment("cch5000", "4-2")
    assert exp.initial_increment == 4
    assert exp.increment == 2
    assert exp.batch_size == 32
    assert exp.official_reported_avg_inc_accuracy == 94.6


def test_lo2ln_command():
    cmd = build_lo2ln_command(
        "external/LO2LN", "eyepacs", "3-1", data_path="data/diabetic", device="0"
    )
    s = command_to_string(cmd)
    assert "-m inclearn" in s
    assert "--initial-increment 3" in s
    assert "--increment 1" in s
    assert "options/der/derE3_diabetic.yaml" in s
