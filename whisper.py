from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pickle

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "ro", split="train+validation", token="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")
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "ro", split="test", token="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")
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"])

print(common_voice)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="ro", task="transcribe")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

# print(common_voice)
#print(common_voice['test'][0])
with open(r"C:\Users\sarbu\OneDrive\Desktop\whisper\common_voice.pkl", "wb") as file:
    pickle.dump(common_voice, file)
