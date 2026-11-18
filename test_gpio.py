import unittest
from unittest.mock import Mock, call, patch

import app


class StopWorker(BaseException):
    pass


class GpioTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = app.state.copy()
        self.saved_generation = app._countdown_generation
        app.state["emergency_trigger"] = False
        self.gpio = Mock(BOARD=10, IN=1, OUT=0, LOW=0, HIGH=1)
        self.gpio.input.return_value = self.gpio.HIGH
        self.patches = [
            patch.object(app, "GPIO", self.gpio),
            patch.object(app, "USE_GPIO", True),
            patch.object(app, "HAS_GPIO", True),
            patch.object(app, "log"),
            patch.object(app, "broadcast_state"),
            patch.object(app, "send_light"),
            patch.object(app.time, "sleep", side_effect=StopWorker),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        app.state.clear()
        app.state.update(self.saved_state)
        app._countdown_generation = self.saved_generation

    def run_one_sample(self):
        with self.assertRaises(StopWorker):
            app._gpio_worker()

    def test_disabled_gpio_does_not_touch_hardware(self):
        with patch.object(app, "USE_GPIO", False):
            app._gpio_worker()
        self.assertEqual(self.gpio.mock_calls, [])

    def test_configured_pins_match_confirmed_wiring(self):
        self.assertEqual(app.PIN_IN, 22)
        self.assertEqual(app.PIN_OUTS, [16])
        self.run_one_sample()
        self.gpio.setmode.assert_called_once_with(self.gpio.BOARD)
        self.assertEqual(self.gpio.setup.call_args_list, [
            call(22, self.gpio.IN),
            call(16, self.gpio.OUT, initial=self.gpio.LOW),
        ])
        self.gpio.input.assert_called_once_with(22)
        self.gpio.output.assert_called_once_with(16, self.gpio.LOW)
        self.assertEqual(app.state["gpio_status"], "enabled")
        app.send_light.assert_not_called()

    def test_failed_relay_initialization_is_not_reported_as_enabled(self):
        self.gpio.setup.side_effect = [None, RuntimeError("relay pin busy")]
        app._gpio_worker()
        self.assertEqual(app.state["gpio_status"], "unavailable")
        self.gpio.input.assert_not_called()
        self.gpio.output.assert_not_called()
        app.send_light.assert_not_called()

    def test_missing_gpio_library_does_not_touch_hardware(self):
        with patch.object(app, "HAS_GPIO", False):
            app._gpio_worker()
        self.assertEqual(app.state["gpio_status"], "unavailable")
        self.assertEqual(self.gpio.mock_calls, [])

    def test_pressed_button_activates_high_level_relay_and_emergency(self):
        self.gpio.input.return_value = self.gpio.LOW
        self.run_one_sample()
        self.gpio.output.assert_called_once_with(16, self.gpio.HIGH)
        self.assertTrue(app.state["emergency_trigger"])
        self.assertEqual(app.state["level"], 4)
        self.assertEqual(app.state["brightness"], 100)
        app.send_light.assert_called_once_with(100, "emergency-on", group_id=6)

    def test_released_button_deactivates_high_level_relay(self):
        app.state["emergency_trigger"] = True
        self.run_one_sample()
        self.gpio.output.assert_called_once_with(16, self.gpio.LOW)
        self.assertFalse(app.state["emergency_trigger"])
        app.send_light.assert_called_once()


if __name__ == "__main__":
    unittest.main()
