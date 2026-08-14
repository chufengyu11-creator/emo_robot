import pytest

from emo_robot_agent.runtime_config import RuntimeModes


@pytest.mark.parametrize(
    ("motion", "tts", "real_motion", "real_tts"),
    [
        ("simulation", "simulation", False, False),
        ("simulation", "real", False, True),
        ("real", "simulation", True, False),
        ("real", "real", True, True),
    ],
)
def test_all_runtime_mode_combinations(motion, tts, real_motion, real_tts):
    modes = RuntimeModes.create(motion, tts, False, "")
    assert modes.real_motion is real_motion
    assert modes.real_tts is real_tts


def test_simulation_motion_rejects_enabled_flag():
    with pytest.raises(ValueError, match="motion_enabled must be false"):
        RuntimeModes.create("simulation", "real", True, "I_UNDERSTAND")


def test_armed_real_motion_requires_exact_confirmation():
    with pytest.raises(ValueError, match="I_UNDERSTAND"):
        RuntimeModes.create("real", "real", True, "")
    modes = RuntimeModes.create("real", "real", True, "I_UNDERSTAND")
    assert modes.motion_enabled


@pytest.mark.parametrize(
    ("motion", "tts"),
    [("invalid", "real"), ("simulation", "invalid")],
)
def test_invalid_mode_is_rejected(motion, tts):
    with pytest.raises(ValueError):
        RuntimeModes.create(motion, tts, False, "")
