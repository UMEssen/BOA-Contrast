import unittest


class BasicTests(unittest.TestCase):
    @staticmethod
    def testImports() -> None:
        from boa_contrast import compute_segmentation, predict  # noqa: F401


if __name__ == "__main__":
    unittest.main()
