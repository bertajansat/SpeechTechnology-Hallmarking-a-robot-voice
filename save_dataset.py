from datasets import load_dataset
import soundfile as sf
import os

# List of audio names you want to download (use names from the dataset)
desired_names = ["MSP-PODCAST_0001_0008.wav","MSP-PODCAST_0001_0019.wav","MSP-PODCAST_0002_0059.wav","MSP-PODCAST_0003_0271.wav","MSP-PODCAST_0006_0029.wav","MSP-PODCAST_0006_0020.wav","MSP-PODCAST_0006_0227.wav","MSP-PODCAST_0008_0067.wav","MSP-PODCAST_0009_0346.wav","MSP-PODCAST_0012_0123.wav","MSP-PODCAST_0014_0022.wav","MSP-PODCAST_0014_0211.wav","MSP-PODCAST_0020_0189.wav","MSP-PODCAST_0020_0213.wav","MSP-PODCAST_0023_0021.wav","MSP-PODCAST_0002_0039.wav","MSP-PODCAST_0024_0503.wav","MSP-PODCAST_0024_0469.wav","MSP-PODCAST_0043_0096.wav","MSP-PODCAST_0047_0263.wav","MSP-PODCAST_0047_0433.wav","MSP-PODCAST_0047_0447.wav","MSP-PODCAST_0051_0797.wav","MSP-PODCAST_0052_0051.wav","MSP-PODCAST_0052_0094.wav","MSP-PODCAST_4515_0062_003.wav","MSP-PODCAST_4529_0064.wav","MSP-PODCAST_4533_0026_000.wav","MSP-PODCAST_4563_0050.wav","MSP-PODCAST_4552_0193.wav","MSP-PODCAST_4552_0512_0000.wav", "MSP-PODCAST_4546_0057_0000.wav", "MSP-PODCAST_4644_0109_0002.wav","PODCAST_4644_0157_000.wav"]  # Replace with actual audio names

# Streaming → only download what's needed
dataset = load_dataset(
    "AbstractTTS/PODCAST",
    split="train",
    streaming=True
)
print("Creating folder...")
# Create the folder to save the audios if it doesn't exist
os.makedirs("Podcast_dataset", exist_ok=True)

# Download only the desired audios by name
print("Downloading audios...")
for example in dataset:
    # Extract the audio name from the 'path' in the dataset (audio file name)
    audio_name = example["audio"]["path"].split("/")[-1].replace(".flac", ".wav")  # Assuming the format is .flac
    if audio_name in desired_names:
        print(f"Saving audio file {audio_name}...")
        audio = example["audio"]
        # Save the audio with the original name in the Podcast_dataset folder
        file_name = f"Podcast_dataset/{audio_name}"
        sf.write(file_name, audio["array"], audio["sampling_rate"])
        print(f"Audio {audio_name} saved as {file_name}")
