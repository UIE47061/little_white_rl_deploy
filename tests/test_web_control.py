"""Safety and command semantics, without DDS, GUI, or a robot."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pathlib import Path

from little_white_wtw import Controller
from web.control import WebControl
from wtw_commands import WTWCommandState, validate_command


@pytest.fixture
def controller():
    cfg = yaml.safe_load((Path(__file__).parents[1]/"little_white_wtw.yaml").read_text())
    runtime = SimpleNamespace(config=cfg, commands=np.array(cfg["cmd_init"],dtype=np.float32),
                              default=np.array(cfg["default_angles"],dtype=np.float32),
                              motor_to_policy=np.arange(12),
                              motor_joints=[f"{leg}_{part}_joint" for leg in cfg["motor_leg_order"]
                                            for part in ("hip","thigh","calf")],
                              reset=lambda:None, validate_commands=validate_command)
    c = Controller(runtime, use_gui=False)
    c.running = True
    c.state = (np.zeros(12,dtype=np.float32),np.zeros(12,dtype=np.float32),np.array([1.,0,0,0]))
    c.state_time = time.monotonic()
    c.web_control = WebControl(c)
    return c


def request(c, message, client="phone", received=None):
    now=time.monotonic()
    c.web_control.handle(client,message,now if received is None else received,now)


def arm(c):
    request(c,{"type":"acquire"})
    request(c,{"type":"mode","mode":"stand"})
    c.transition_start-=c.config["transition_seconds"]
    request(c,{"type":"mode","mode":"move"})


def move(c,**values):
    request(c,{"type":"control","epoch":c.web_control.epoch,
               "vx":0.1,"vy":0.1,"yaw_rate":0.1,**values})


def test_start_is_damping_and_move_requires_completed_stand(controller):
    c=controller
    assert c.mode=="stop"
    request(c,{"type":"acquire"})
    with pytest.raises(ValueError): request(c,{"type":"mode","mode":"move"})
    request(c,{"type":"mode","mode":"stand"})
    with pytest.raises(ValueError): request(c,{"type":"mode","mode":"move"})
    c.transition_start-=3
    request(c,{"type":"mode","mode":"move"})
    assert c.mode=="move" and np.all(c.runtime.commands[:3]==0)


def test_second_client_cannot_mutate_or_release_owner(controller):
    c=controller;arm(c);move(c)
    for message in ({"type":"acquire"},{"type":"mode","mode":"stop"},
                    {"type":"parameters","values":{"height":.02}}):
        with pytest.raises(ValueError):request(c,message,"viewer")
    request(c,{"type":"release"},"viewer")
    assert c.web_control.owner=="phone" and c.mode=="move"


def test_timeout_zeroes_returns_stand_and_invalidates_session(controller):
    c=controller;arm(c);move(c)
    c.web_control.last_heartbeat-=.31
    c.web_control.watchdog()
    assert c.mode=="stand" and c.web_control.owner is None
    assert np.all(c.runtime.commands[:3]==0)
    with pytest.raises(ValueError):move(c)
    request(c,{"type":"acquire"})
    assert c.mode=="stand"


def test_stale_dds_takes_priority_over_return_to_stand(controller):
    c=controller;arm(c);move(c)
    c.state_time-=.6
    c.web_control.last_heartbeat-=.31
    c.web_control.watchdog()
    assert c.mode=="stop" and c.web_control.owner is None
    assert np.all(c.runtime.commands[:3]==0)


@pytest.mark.parametrize("mode",["sit","stand","stop"])
def test_release_preserves_non_move_modes(controller,mode):
    c=controller;request(c,{"type":"acquire"});request(c,{"type":"mode","mode":mode})
    request(c,{"type":"release"})
    assert c.mode==mode


def test_invalid_parameter_batch_is_atomic_and_history_is_preserved(controller):
    c=controller;arm(c)
    resets=[];c.runtime.reset=lambda:resets.append(True)
    before=c.runtime.commands.copy()
    with pytest.raises(ValueError):
        request(c,{"type":"parameters","values":{"height":.02,"frequency":100}})
    np.testing.assert_array_equal(c.runtime.commands,before)
    request(c,{"type":"parameters","values":{"height":.02,"frequency":3.0}})
    request(c,{"type":"gait","gait":"pace"})
    assert np.isclose(c.runtime.commands[3],.02) and not resets
    assert c.mode=="move"


@pytest.mark.parametrize("bad",[float('nan'),float('inf'),10**1000,"0.1",True,[],None])
def test_bad_input_does_not_mutate_velocity(controller,bad):
    c=controller;arm(c)
    with pytest.raises(ValueError):move(c,vx=bad)
    assert np.all(c.runtime.commands[:3]==0)


def test_old_epoch_and_delayed_packets_cannot_restart_motion(controller):
    c=controller;arm(c);epoch=c.web_control.epoch
    request(c,{"type":"mode","mode":"stop"})
    with pytest.raises(ValueError):request(c,{"type":"control","epoch":epoch,"vx":.2,"vy":0,"yaw_rate":0})
    with pytest.raises(ValueError):request(c,{"type":"mode","mode":"stand"},received=time.monotonic()-.4)
    assert c.mode=="stop" and np.all(c.runtime.commands[:3]==0)


def test_presets_are_staged_and_keyboard_web_limits_match(controller):
    c=controller;request(c,{"type":"acquire"})
    request(c,{"type":"parameters","values":{"height":.02}})
    assert c.mode=="stop" and np.isclose(c.runtime.commands[3],.02)
    keyboard=WTWCommandState(c.web_control.defaults,c.config.get("wtw_keyboard_limits"))
    for spec in c.web_control.config["parameters"].values():
        assert any(i==spec["index"] and lo==spec["min"] and hi==spec["max"]
                   for i,step,lo,hi in keyboard.adjustments.values())
    with pytest.raises(ValueError):request(c,{"type":"parameters","values":{"idle_step":1}})


def test_controller_loop_watchdog_works_without_web_bridge_thread(controller):
    c=controller;arm(c);move(c)
    c.web_control.last_heartbeat-=.31
    published=[]
    def publish(target=None):
        published.append(c.mode)
        c.running=False
    c._publish=publish
    c.LowCmdWrite()
    assert published[0]=="stand" and c.web_control.owner is None
