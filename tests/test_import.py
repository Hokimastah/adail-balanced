import adail
from adail.data import ClassIncrementalScenario, get_preset


def test_version():
    assert adail.__version__ == "0.5.0"


def test_proposal_preset():
    preset = get_preset("eyepacs")
    assert preset.num_classes == 5
    assert preset.ordinal is True


def test_scenario_from_preset():
    scenario = ClassIncrementalScenario.from_preset("cch5000", "4-2")
    assert scenario.tasks[0] == [0, 1, 2, 3]
    assert scenario.tasks[1] == [4, 5]
