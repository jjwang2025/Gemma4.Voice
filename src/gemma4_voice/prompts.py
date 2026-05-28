"""Prompt templates for Gemma 4 speech transcription."""

ASR_PROMPT = (
    "Transcribe the following speech segment in English into English text.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."
)

TIMESTAMP_ASR_PROMPT = (
    "Transcribe the following speech segment in English into English text and include timestamps.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Split the transcription into natural speech segments.\n"
    "* Format each line exactly as [HH:MM:SS.mmm --> HH:MM:SS.mmm] text.\n"
    "* Use timestamps relative to the start of the audio.\n"
    "* Output only the timestamped transcription."
)


def prompt_for_mode(mode: str) -> str:
    if mode == "timestamps":
        return TIMESTAMP_ASR_PROMPT
    return ASR_PROMPT
