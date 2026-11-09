"""Reproduce mode/heartbeat races with deterministic times and no DDS connection."""
import unittest
from unittest.mock import patch
import numpy as np
from little_white_wtw import WTWPolicy, Controller
from web.control import WebControl

class WebModeLeaseTest(unittest.TestCase):
    def setUp(self):
        self.c=Controller(WTWPolicy(),use_gui=False)
        self.c.running=True
        self.c.mode='stand'
        self.c.transition_start=96.
        self.c.state=(self.c.target.copy(),np.zeros(12),np.array([1.,0.,0.,0.]))
        self.c.state_time=100.
        self.web=WebControl(self.c)
        self.web.owner='a';self.web.last_heartbeat=100.

    def send(self,msg,t,client='a'):
        self.c.state_time=t
        with patch('time.monotonic',return_value=t):
            self.web.handle(client,msg,t,t)

    def move(self):
        self.send({'type':'mode','mode':'move'},100.29)

    def test_move_near_expiry_can_receive_new_epoch_control(self):
        self.move()
        self.send(dict(type='control',epoch=self.web.epoch,vx=0,vy=0,yaw_rate=0),100.34)
        self.assertEqual(self.c.mode,'move')
        self.assertEqual(self.web.owner,'a')

    def test_missing_control_still_returns_to_stand(self):
        self.move();self.c.state_time=100.61
        with patch('time.monotonic',return_value=100.61):self.web.watchdog(100.61)
        self.assertEqual(self.c.mode,'stand')
        self.assertIsNone(self.web.owner)
        np.testing.assert_array_equal(self.c.runtime.commands[:3],0)

    def test_stale_epoch_is_rejected_without_cancelling_mode(self):
        old=self.web.epoch;self.move()
        with self.assertRaisesRegex(ValueError,'Mode changed'):
            self.send(dict(type='control',epoch=old,vx=.3,vy=0,yaw_rate=0),100.34)
        self.assertEqual(self.c.mode,'move')
        np.testing.assert_array_equal(self.c.runtime.commands[:3],0)
        self.send(dict(type='control',epoch=self.web.epoch,vx=.2,vy=0,yaw_rate=0),100.39)
        self.assertAlmostEqual(self.c.runtime.commands[0],.2)

    def test_invalid_mode_or_other_client_cannot_renew_lease(self):
        with self.assertRaises(ValueError):self.send({'type':'mode','mode':'fly'},100.29)
        self.assertEqual(self.web.last_heartbeat,100.)
        with self.assertRaises(ValueError):self.send({'type':'mode','mode':'move'},100.29,'other')
        self.assertEqual(self.web.last_heartbeat,100.)

    def test_explicit_release_explains_focus_loss(self):
        self.move();self.send({'type':'release','reason':'window_blur'},100.34)
        self.assertEqual(self.c.mode,'stand')
        self.assertIn('失去焦點',self.web.reason)

    def test_dds_stale_still_goes_to_damping(self):
        self.move();self.c.state_time=99.
        with patch('time.monotonic',return_value=100.35):self.web.watchdog(100.35)
        self.assertEqual(self.c.mode,'stop')
        self.assertIsNone(self.web.owner)

    def test_nonstring_release_reason_still_releases(self):
        self.move();self.send({'type':'release','reason':[]},100.34)
        self.assertEqual(self.c.mode,'stand')
        self.assertIsNone(self.web.owner)
