import unittest

from gemma4_voice.prompts import ASR_PROMPT, TIMESTAMP_ASR_PROMPT, prompt_for_mode


class PromptSelectionTest(unittest.TestCase):
    def test_prompt_for_transcribe(self) -> None:
        self.assertEqual(prompt_for_mode("transcribe"), ASR_PROMPT)

    def test_prompt_for_timestamps(self) -> None:
        self.assertEqual(prompt_for_mode("timestamps"), TIMESTAMP_ASR_PROMPT)


if __name__ == "__main__":
    unittest.main()
