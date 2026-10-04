import socket
import struct
import threading
import unittest

import cv2
import numpy as np

from dex_teleop.frame_transport import CameraFrameServer


class FrameTransportTests(unittest.TestCase):
    def test_receives_length_prefixed_jpeg_frame(self):
        server = CameraFrameServer("127.0.0.1", 0)
        host, port = server.address
        image = np.full((24, 32, 3), (10, 80, 200), dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(encoded_ok)
        payload = encoded.tobytes()

        def send_frame():
            with socket.create_connection((host, port), timeout=2) as client:
                client.sendall(struct.pack("!I", len(payload)) + payload)

        sender = threading.Thread(target=send_frame)
        sender.start()
        try:
            received = server.read()
            self.assertEqual(received.shape, image.shape)
            self.assertTrue(np.allclose(received, image, atol=10))
        finally:
            server.close()
            sender.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
