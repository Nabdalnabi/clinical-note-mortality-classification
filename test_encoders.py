import unittest
from encoders import ENCODERS, specification


class EncoderTests(unittest.TestCase):
    def test_four_models(self):
        self.assertEqual(len(ENCODERS), 4)
        self.assertEqual(len({v[1] for v in ENCODERS.values()}), 4)

    def test_window_configuration(self):
        for name in ENCODERS:
            _, config = specification(name)
            self.assertGreater(config['chunk_batch_size'], 0)
            self.assertLess(config['stride'], config['max_length'] - 2)
        self.assertEqual(specification('clinical-longformer')[1]['max_length'], 4096)


if __name__ == '__main__':
    unittest.main()
