import unittest
from backend.taco_adapter import dataset_status


class TacoAdapterTests(unittest.TestCase):
    def test_official_annotations_are_available(self):
        status = dataset_status()
        self.assertTrue(status["available"])
        self.assertGreater(status["images"], 1_000)
        self.assertGreater(status["annotations"], 4_000)
