from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gemma4_voice.batch import discover_audio_files


class BatchDiscoveryTest(unittest.TestCase):
    def test_discover_audio_files_non_recursive(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.wav").write_text("x", encoding="utf-8")
            (root / "b.txt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.wav").write_text("x", encoding="utf-8")

            files = discover_audio_files(root, recursive=False)
            self.assertEqual([path.name for path in files], ["a.wav"])

    def test_discover_audio_files_recursive(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.wav").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.flac").write_text("x", encoding="utf-8")

            files = discover_audio_files(root, recursive=True)
            self.assertEqual([path.name for path in files], ["a.wav", "c.flac"])


if __name__ == "__main__":
    unittest.main()
