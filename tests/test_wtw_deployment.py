import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from little_white_mjlab import Controller as MJLabController
from little_white_wtw import Controller, DEFAULT_CONFIG, WTWPolicy
from web.control import make_config


class WTWDeploymentTest(unittest.TestCase):
    def test_zero_velocity_keeps_gait_phase_and_policy_actions_active(self):
        runtime = WTWPolicy()
        runtime.commands[:3] = 0
        q = runtime.default[runtime.motor_to_policy]
        targets = []
        for _ in range(10):
            targets.append(runtime.infer(q, np.zeros(12), [1, 0, 0, 0]))
        self.assertGreater(float(runtime.phase), 0.4)
        self.assertGreater(float(np.ptp(np.asarray(targets), axis=0).max()), 1e-3)
        self.assertGreater(np.count_nonzero(np.any(runtime.history, axis=1)), 1)

    def test_stopping_preserves_body_commands_and_resets_history(self):
        runtime = WTWPolicy()
        controller = Controller(runtime, use_gui=False)
        runtime.commands[0] = 0.2
        q = runtime.default[runtime.motor_to_policy]
        runtime.infer(q, np.zeros(12), [1, 0, 0, 0])
        body = runtime.commands[3:].copy()
        self.assertTrue(np.any(runtime.history))
        controller.set_mode('stop')
        np.testing.assert_array_equal(runtime.commands[:3], np.zeros(3))
        np.testing.assert_array_equal(runtime.commands[3:], body)
        self.assertFalse(np.any(runtime.history))
        self.assertEqual(runtime.phase, 0)
        self.assertTrue(np.isfinite(runtime.infer(q, np.zeros(12), [1, 0, 0, 0])).all())

    def test_console_changes_preserve_history_and_reject_out_of_range_velocity(self):
        runtime = WTWPolicy()
        controller = Controller(runtime, use_gui=False)
        runtime.infer(runtime.default[runtime.motor_to_policy], np.zeros(12), [1, 0, 0, 0])
        history, phase = runtime.history.copy(), runtime.phase
        controller.command('cmd vx 0.2')
        controller.command('gait pace')
        self.assertAlmostEqual(runtime.commands[0], 0.2)
        np.testing.assert_array_equal(runtime.commands[5:8], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(runtime.history, history)
        self.assertEqual(runtime.phase, phase)
        with self.assertRaises(ValueError):
            controller.command(f"cmd vx {runtime.config['max_lin'] + .1}")
        self.assertAlmostEqual(runtime.commands[0], 0.2)
        with self.assertRaises(ValueError):
            controller.command('ctl idle_step 1')

    def test_wrong_pd_digest_or_widened_velocity_limits_are_rejected(self):
        original = yaml.safe_load(DEFAULT_CONFIG.read_text())
        for override in ({'kps': [20.0] * 12}, {'policy_sha256': '0' * 64}, {'max_lin': original['max_lin'] + .1}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as folder:
                cfg = copy.deepcopy(original)
                cfg.update(override)
                for key in ('policy_path', 'metadata_path'):
                    cfg[key] = str(DEFAULT_CONFIG.parent / cfg[key])
                path = Path(folder) / 'config.yaml'
                path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    WTWPolicy(path)

    def test_web_uses_new_limits_and_disables_unsupported_idle_control(self):
        config = make_config(WTWPolicy())
        self.assertEqual(config['speeds'], dict(forward=yaml.safe_load(DEFAULT_CONFIG.read_text())['max_lin'], backward=.2, lateral=.15, yaw=.5))
        self.assertFalse(config['idle_control'])

    def test_forward_acceleration_on_move_and_release(self):
        runtime = WTWPolicy()
        runtime.forward_accel = .8
        runtime.commands[0] = runtime.config["max_lin"]
        runtime.reset()
        runtime.prepare_command()
        self.assertAlmostEqual(runtime.effective_commands[0], .8 * runtime.control_dt)
        runtime.commands[0] = 0.
        runtime.prepare_command()
        self.assertEqual(runtime.effective_commands[0], 0.)

    def test_mjlab_entry_remains_loadable(self):
        c = MJLabController(use_gui=False)
        target = c.infer(c.default_angles[c.motor_to_policy], np.zeros(12), [1, 0, 0, 0])
        self.assertEqual(c.obs.shape, (45,))
        self.assertTrue(np.isfinite(target).all())


if __name__ == '__main__':
    unittest.main()
