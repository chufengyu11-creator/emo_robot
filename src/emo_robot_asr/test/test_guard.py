from emo_robot_asr.guard import AgentGuard


def test_planner_guard_expires_when_no_agent_status_arrives():
    now = [10.0]
    guard = AgentGuard(20.0, 1.0, clock=lambda: now[0])
    guard.text_published()
    assert guard.blocked()
    now[0] = 30.0
    assert not guard.blocked()


def test_running_blocks_until_terminal_then_cools_down():
    now = [10.0]
    guard = AgentGuard(20.0, 1.0, clock=lambda: now[0])
    guard.text_published()
    assert not guard.agent_state("running")
    now[0] = 100.0
    assert guard.blocked()
    assert guard.agent_state("plan_succeeded")
    assert guard.blocked()
    now[0] = 101.0
    assert not guard.blocked()


def test_plan_failure_is_also_terminal():
    now = [1.0]
    guard = AgentGuard(20.0, 0.0, clock=lambda: now[0])
    guard.agent_state("running")
    assert guard.agent_state("plan_failed")
    assert not guard.blocked()
