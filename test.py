import pickle
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

with open(r"C:\Users\sarbu\OneDrive\Desktop\whisper\common_voice.pkl", "rb") as file:
    common_voice = pickle.load(file)
print(common_voice)

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="ro", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="ro", task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

major, _ = torch.cuda.get_device_capability()
if major >= 6:
    print("Your GPU supports FP16")
else:
    print("Your GPU does not support FP16")

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ro",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    # fsdp=["full_shard", "auto_wrap"],
    # fsdp_transformer_layer_cls_to_wrap="T5Block",
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="whisper_small_ro_VladS_test",
    hub_token="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",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
trainer.train()
print("we got here..")
trainer.push_to_hub()

# # kwargs = {
# #     "dataset_tags": "mozilla-foundation/common_voice_13_0",
# #     "dataset": "Common Voice 13.0",  # a 'pretty' name for the training dataset
# #     "dataset_args": "config: ro, split: test",
# #     "language": "ro",
# #     "model_name": "Whisper Small ro - VladS",  # a 'pretty' name for our model
# #     "finetuned_from": "openai/whisper-small",
# #     "tasks": "automatic-speech-recognition",
# #     "tags": "hf-asr-leaderboard",
# # }
# # trainer.push_to_hub(**kwargs)