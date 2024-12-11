import wave
import os
import pyaudio
from faster_whisper import WhisperModel

# Define constants for colors (you may modify these as needed)
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"


# Function to record a chunk of audio
def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


# Function to transcribe a chunk using Faster Whisper
def transcribe_chunk(model, chunk_file):
    """
    Transcribe the audio chunk using the Whisper model.

    Args:
        model: The Faster Whisper model instance.
        chunk_file: Path to the audio chunk file to be transcribed.

    Returns:
        Transcription (string) of the audio.
    """
    segments, _ = model.transcribe(chunk_file, beam_size=5)  # Adjust beam_size as needed
    transcription = " ".join(segment.text for segment in segments)
    return transcription


# Main function
def main2():
    # Choose your model settings
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = ""  # Initialize an empty string to accumulate transcriptions

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(NEON_GREEN + transcription + RESET_COLOR)

            os.remove(chunk_file)  # Clean up the temporary chunk file

            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")
        # Write the accumulated transcription to the log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG: " + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main2()
