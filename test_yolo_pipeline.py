import unittest
from unittest.mock import Mock, patch

import numpy as np

import app


class StopWorker(BaseException):
    pass


class YoloPipelineTests(unittest.TestCase):
    def setUp(self):
        self.saved_state = app.state.copy()
        self.saved_camera = app.global_camera
        self.saved_jpeg = app.latest_jpeg
        self.saved_raw = app.latest_raw_frame
        app.global_camera = None

    def tearDown(self):
        app.state.clear()
        app.state.update(self.saved_state)
        app.global_camera = self.saved_camera
        app.latest_jpeg = self.saved_jpeg
        app.latest_raw_frame = self.saved_raw

    def test_model_loading_defers_fusion_to_prediction_backend(self):
        with patch.object(app, "YOLO") as yolo, patch.object(app, "log"):
            model = app.load_yolo_model()
        yolo.assert_called_once_with(app.YOLO_MODEL)
        self.assertIs(model, yolo.return_value)
        model.fuse.assert_not_called()
        model.assert_not_called()

    def test_black_image_is_reported_as_dark(self):
        status, luma = app.camera_image_health(np.zeros((544, 960, 3), dtype=np.uint8))
        self.assertEqual(status, "dark")
        self.assertEqual(luma, 0.0)

    def test_normal_and_dim_images_are_not_reported_as_black(self):
        for brightness in (20, 128, 255):
            with self.subTest(brightness=brightness):
                frame = np.full((544, 960, 3), brightness, dtype=np.uint8)
                status, luma = app.camera_image_health(frame)
                self.assertEqual(status, "ok")
                self.assertEqual(luma, brightness)

    def test_worker_failure_reports_error_and_releases_camera(self):
        camera = Mock()
        app.global_camera = camera
        app.latest_jpeg = b"stale frame"
        app.state["camera_status"] = "connected"
        app.state["camera_image_status"] = "ok"
        app.state["pipeline_fps"] = 15.0
        with patch.object(app, "_camera_pipeline", side_effect=RuntimeError("test failure")), \
                patch.object(app, "log"), patch.object(app, "broadcast_state"), \
                patch.object(app.time, "sleep", side_effect=StopWorker):
            with self.assertRaises(StopWorker):
                app.camera_worker()
        self.assertEqual(app.state["yolo_status"], "error")
        self.assertIn("test failure", app.state["yolo_error"])
        self.assertEqual(app.state["camera_status"], "disconnected")
        self.assertEqual(app.state["camera_image_status"], "unknown")
        self.assertEqual(app.state["pipeline_fps"], 0.0)
        self.assertIsNone(app.latest_jpeg)
        self.assertIsNone(app.global_camera)
        camera.stop.assert_called_once_with()

    def test_worker_retries_after_failure(self):
        with patch.object(app, "_camera_pipeline", side_effect=[RuntimeError("retry"), StopWorker]) as pipeline, \
                patch.object(app, "log"), patch.object(app, "broadcast_state"), \
                patch.object(app.time, "sleep") as sleep:
            with self.assertRaises(StopWorker):
                app.camera_worker()
        self.assertEqual(pipeline.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertEqual(app.state["yolo_status"], "loading")
        self.assertEqual(app.state["yolo_error"], "")

    def test_status_api_exposes_detection_health(self):
        with app.app.test_client() as client:
            response = client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        for key in ("camera_image_status", "camera_mean_luma", "yolo_status", "yolo_error", "yolo_last_success"):
            self.assertIn(key, response.get_json())

    def test_page_renders_detection_health_fields(self):
        with app.app.test_client() as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="camera-health"', response.data)
        self.assertIn(b'id="yolo-health"', response.data)


if __name__ == "__main__":
    unittest.main()
