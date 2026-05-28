from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gemma4_voice.subtitles import default_output_paths, format_srt_timestamp, timestamp_text_to_srt


class SubtitleHelpersTest(unittest.TestCase):
    def test_format_srt_timestamp(self) -> None:
        self.assertEqual(format_srt_timestamp("00:00:05.120"), "00:00:05,120")

    def test_timestamp_text_to_srt(self) -> None:
        text = (
            "[00:00:00.000 --> 00:00:05.120] Hello there.\n"
            "[00:00:05.120 --> 00:00:06.000] Bye."
        )
        expected = (
            "1\n00:00:00,000 --> 00:00:05,120\nHello there.\n\n"
            "2\n00:00:05,120 --> 00:00:06,000\nBye.\n"
        )
        self.assertEqual(timestamp_text_to_srt(text), expected)

    def test_default_output_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "sample.wav"
            text_path, srt_path = default_output_paths(audio, "q4", "timestamps")
            self.assertEqual(text_path.name, "sample.q4.timestamps.txt")
            self.assertEqual(srt_path.name, "sample.q4.timestamps.srt")


if __name__ == "__main__":
    unittest.main()
