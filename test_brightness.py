import copy
import unittest
from unittest.mock import Mock, patch

import app


MAC_A = "0024E979D6"
MAC_B = "0024E9D1FF"
ACK_A = bytes.fromhex("0211370024E979D6011600001E6400003F")
ACK_B = bytes.fromhex("0211370024E9D1FF011600001E3E00009A")
STATUS_A = bytes.fromhex("021C800024E979D60F170482001F3C0064014E014E00003E06020A53")


def change_packet(packet, cmd=None, brightness=None, mac=None):
    result = bytearray(packet)
    if cmd is not None:
        result[2] = cmd
    if brightness is not None:
        result[23 if result[2] == 0x80 else 13] = brightness
    if mac is not None:
        result[3:8] = bytes.fromhex(mac)
    result[-1] = sum(result[:-1]) % 256
    return bytes(result)


class BrightnessTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = copy.deepcopy(app.state)
        self.send_light = app.send_light
        app.state["device_brightness"] = {MAC_A: None, MAC_B: None}
        app.state["device_brightness_ts"] = {}
        app.state["device_detail"] = {MAC_A: {}, MAC_B: {}}
        app.state["device_target_brightness"] = {MAC_A: 10, MAC_B: 10}
        app.state["telemetry_history"] = []
        self.patches = [
            patch.object(app, "log"),
            patch.object(app, "send_light"),
            patch.object(app, "_device_last_seen", {}),
            patch.object(app, "_retry_inflight", {}),
            patch.object(app.time, "time", return_value=1000.0),
        ]
        for item in self.patches:
            item.start()
        self.client = app.socketio.test_client(app.app)
        self.client.get_received()

    def tearDown(self):
        self.client.disconnect()
        for item in reversed(self.patches):
            item.stop()
        app.state.clear()
        app.state.update(self.saved_state)

    def parse(self, packet):
        app._parse_status_packet(packet, "192.168.50.202")
        return self.client.get_received()

    def test_real_ack_publishes_mac_and_brightness_to_both_views(self):
        events = self.parse(ACK_A)
        update = next(event["args"][0] for event in events if event["name"] == "device_brightness_update")
        self.assertEqual(update["mac"], MAC_A)
        self.assertEqual(update["brightness"], 100)
        self.assertEqual(update["ts"], 1000.0)
        detail = next(event["args"][0] for event in events if event["name"] == "device_detail_update")
        self.assertEqual(detail["dimming"], 100)
        self.assertEqual(detail["mac"], MAC_A)
        self.assertTrue(detail["telemetry_stale"])
        snapshot = next(event["args"][0] for event in events if event["name"] == "state")
        self.assertEqual(snapshot["device_brightness"][MAC_A], 100)
        self.assertEqual(snapshot["device_detail"][MAC_A]["dimming"], 100)

    def test_nodes_sharing_an_ip_are_updated_independently_by_mac(self):
        self.parse(ACK_A)
        self.parse(ACK_B)
        self.assertEqual(app.state["device_brightness"], {MAC_A: 100, MAC_B: 62})
        self.assertEqual(app.state["device_detail"][MAC_A]["dimming"], 100)
        self.assertEqual(app.state["device_detail"][MAC_B]["dimming"], 62)

    def test_actual_report_does_not_replace_control_target(self):
        self.parse(ACK_A)
        self.assertEqual(app.state["device_brightness"][MAC_A], 100)
        self.assertEqual(app.state["device_target_brightness"][MAC_A], 10)

    def test_real_telemetry_preserves_full_measurements(self):
        self.parse(STATUS_A)
        detail = app.state["device_detail"][MAC_A]
        self.assertEqual(detail["dimming"], 62)
        self.assertEqual(detail["voltage"], 115.4)
        self.assertEqual(detail["current"], 0.31)
        self.assertFalse(detail["telemetry_stale"])

    def test_ack_updates_dimming_without_discarding_previous_telemetry(self):
        self.parse(STATUS_A)
        self.parse(ACK_A)
        detail = app.state["device_detail"][MAC_A]
        self.assertEqual(detail["dimming"], 100)
        self.assertEqual(detail["voltage"], 115.4)
        self.assertTrue(detail["telemetry_stale"])

    def test_all_supported_short_reports_publish_details_immediately(self):
        for cmd in (0x40, 0x41, 0x53, 0x37):
            with self.subTest(cmd=cmd):
                events = self.parse(change_packet(ACK_A, cmd=cmd, brightness=27))
                detail = next(event["args"][0] for event in events if event["name"] == "device_detail_update")
                self.assertEqual(detail["dimming"], 27)
                self.assertEqual(detail["cmd"], f"0x{cmd:02X}")

    def test_zero_brightness_is_a_valid_report(self):
        self.parse(change_packet(ACK_A, brightness=0))
        self.assertEqual(app.state["device_brightness"][MAC_A], 0)
        self.assertEqual(app.state["device_detail"][MAC_A]["dimming"], 0)

    def test_invalid_brightness_does_not_change_the_display(self):
        events = self.parse(change_packet(ACK_A, brightness=255))
        self.assertIsNone(app.state["device_brightness"][MAC_A])
        self.assertEqual(events, [])

    def test_unknown_mac_does_not_update_other_nodes(self):
        events = self.parse(change_packet(ACK_A, mac="0102030405"))
        self.assertEqual(app.state["device_brightness"], {MAC_A: None, MAC_B: None})
        self.assertEqual(events, [])

    def test_ack_without_brightness_keeps_last_report(self):
        packet = bytes([0x02, 9, 0x37]) + bytes.fromhex(MAC_A) + b"\x00"
        events = self.parse(packet)
        self.assertIsNone(app.state["device_brightness"][MAC_A])
        self.assertNotIn("device_brightness_update", [event["name"] for event in events])

    def test_sending_a_command_does_not_fabricate_a_report(self):
        with patch.object(app, "_start_retry_worker"):
            self.send_light(45, "test", target_device=app.DEVICES[0])
        self.assertEqual(app.state["device_target_brightness"][MAC_A], 45)
        self.assertIsNone(app.state["device_brightness"][MAC_A])

    def test_report_without_a_requested_target_does_not_send_a_command(self):
        app.state["device_target_brightness"] = {MAC_A: None, MAC_B: None}
        self.parse(STATUS_A)
        app.send_light.assert_not_called()
        self.assertEqual(app.state["device_brightness"][MAC_A], 62)

    def test_split_and_concatenated_tcp_packets_both_update_their_nodes(self):
        conn = Mock()
        conn.__enter__ = Mock(return_value=conn)
        conn.__exit__ = Mock(return_value=False)
        conn.recv.side_effect = [STATUS_A[:5], STATUS_A[5:] + ACK_B[:8], ACK_B[8:], b""]
        with patch.object(app, "active_tcp_clients", {"192.168.50.202": conn}):
            app.handle_tcp_client(conn, "192.168.50.202")
        self.assertEqual(app.state["device_brightness"], {MAC_A: 62, MAC_B: 62})


class ControlRetryTests(unittest.TestCase):
    def setUp(self):
        self.device_a, self.device_b = app.DEVICES[:2]
        self.original_state = copy.deepcopy(app.state)
        for name, value in {
            "_retry_inflight": {},
            "_device_last_seen": {},
            "active_tcp_clients": {},
            "log": Mock(),
            "broadcast_state": Mock(),
        }.items():
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.restore_state)

    def restore_state(self):
        app.state.clear()
        app.state.update(self.original_state)

    def test_group_requires_matching_brightness_for_each_configured_light(self):
        members = [self.device_a, self.device_b]
        app._start_retry_worker(
            self.device_a, app.build_group_control_packet(MAC_A, 6, 30), "test", members
        )
        pending = next(iter(app._retry_inflight.values()))
        app._confirm_control(MAC_A, 0x37, 30)
        self.assertEqual(set(pending["devices"]), {MAC_A, MAC_B})

        pending["awaiting"] = {mac: ({0x36}, app.time.monotonic()) for mac in (MAC_A, MAC_B)}
        app._parse_status_packet(change_packet(ACK_A, brightness=0), "192.168.50.202")
        self.assertEqual(set(pending["devices"]), {MAC_A, MAC_B})
        app._parse_status_packet(change_packet(ACK_A, cmd=0x53, brightness=30), "192.168.50.202")
        self.assertEqual(set(pending["devices"]), {MAC_A, MAC_B})

        app._parse_status_packet(change_packet(ACK_A, brightness=30), "192.168.50.202")
        self.assertEqual(set(pending["devices"]), {MAC_B})
        app._parse_status_packet(change_packet(ACK_B, brightness=30), "192.168.50.202")
        self.assertFalse(app._retry_inflight)

    def test_group_six_enqueues_one_original_group_frame(self):
        with patch.object(app, "_start_retry_worker") as enqueue:
            ok, _ = app.send_light(10, "auto", group_id=6)
        self.assertTrue(ok)
        enqueue.assert_called_once()
        representative, packet, reason, members = enqueue.call_args.args
        self.assertEqual(representative["id"], MAC_A)
        self.assertEqual(packet, app.build_group_control_packet(MAC_A, 6, 10))
        self.assertEqual(reason, "auto-group-6")
        self.assertEqual([device["id"] for device in members], [MAC_A, MAC_B])

    def test_late_group_report_gets_a_separate_targeted_retry(self):
        class StopWorker(Exception):
            pass

        gateway = Mock()
        app.active_tcp_clients["192.168.50.202"] = gateway
        app.send_light(30, "test", group_id=6)
        item = app._retry_inflight[(0x36, "6")]
        item["attempts"] = len(app.RETRY_DELAYS)
        item["awaiting"] = {mac: ({0x36}, app.time.monotonic()) for mac in (MAC_A, MAC_B)}
        app._confirm_control(MAC_A, 0x37, 30)

        self.assertEqual(set(item["devices"]), {MAC_B})
        self.assertEqual(item["attempts"], 0)
        self.assertEqual(item["retry_delays"], app.GROUP_FOLLOWUP_DELAYS)
        item["next_send_time"] = 0
        with patch.object(app._retry_ready, "wait", side_effect=StopWorker):
            with self.assertRaises(StopWorker):
                app._central_retry_worker()
        gateway.sendall.assert_called_once_with(app.build_light_packet(MAC_B, 30))
        app._confirm_control(MAC_B, 0x37, 30)
        self.assertFalse(app._retry_inflight)

    def test_emergency_blocks_web_and_rest_control_but_allows_emergency_dimming(self):
        app.state.update(mode="manual", emergency_trigger=True, level=4, brightness=100)
        client = app.socketio.test_client(app.app)
        with patch.object(app, "save_devices") as save:
            client.emit("set_mode", {"mode": "auto"})
            client.emit("set_brightness", {"brightness": 9})
            client.emit("send_brightness", {"device_id": MAC_A, "brightness": 9})
            client.emit("set_group", {"device_id": MAC_A, "group_id": 7})
            save.assert_not_called()
        client.disconnect()
        with app.app.test_client() as web:
            self.assertEqual(web.post("/api/light", json={"brightness": 9}).status_code, 409)
            self.assertEqual(web.post("/api/light/" + MAC_A, json={"brightness": 9}).status_code, 409)
        self.assertEqual((app.state["mode"], app.state["level"], app.state["brightness"]), ("manual", 4, 100))
        self.assertFalse(app._retry_inflight)
        self.assertFalse(app.send_light(9, "queued-before-emergency")[0])
        self.assertTrue(app.send_light(100, "emergency-on", group_id=6)[0])
        emergency = app._retry_inflight[(0x36, "6")]
        self.assertEqual(emergency["packet"][-2], 100)

    def test_new_single_target_wins_over_an_older_group_send(self):
        app._start_retry_worker(self.device_a, app.build_group_control_packet(MAC_A, 6, 30),
                                "group", [self.device_a, self.device_b])
        app._start_retry_worker(self.device_a, app.build_light_packet(MAC_A, 60), "single")
        group = app._retry_inflight[(0x36, "6")]
        single = app._retry_inflight[(0x52, MAC_A)]
        self.assertEqual(set(group["devices"]), {MAC_B})
        self.assertEqual(set(single["devices"]), {MAC_A})

    def test_old_socket_cleanup_leaves_replacement_connected(self):
        old = Mock()
        old.__enter__ = Mock(return_value=old)
        old.__exit__ = Mock(return_value=False)
        old.recv.return_value = b""
        replacement = object()
        app.active_tcp_clients["192.168.50.202"] = replacement
        app.handle_tcp_client(old, "192.168.50.202")
        self.assertIs(app.active_tcp_clients["192.168.50.202"], replacement)

    def test_unchanged_people_count_commits_upgrade_at_stability_deadline(self):
        app.state.update(mode="auto", emergency_trigger=False, level=1, lux_avg=0)
        with patch.object(app, "_auto_candidate_level", None), \
                patch.object(app, "_auto_candidate_since", 0.0), \
                patch.object(app, "AUTO_LEVEL_STABLE_SEC", 1.0), \
                patch.object(app.time, "monotonic", side_effect=[10.0, 10.5, 11.0]), \
                patch.object(app, "send_light") as send:
            app.set_level_from_count(1)
            app.set_level_from_count(1)
            send.assert_not_called()
            app.set_level_from_count(1)
            send.assert_called_once()
        self.assertEqual(app.state["level"], 2)


if __name__ == "__main__":
    unittest.main()
