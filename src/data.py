import torch
import whisper
from datasets import load_dataset, Audio
from torch.nn.utils.rnn import pad_sequence

def extract_whisper_features_from_array(whisper_model, wav_array, device):
    """extract whisper features from audio array"""
    audio = torch.from_numpy(wav_array).float()
    
    # pad/trim the raw audio to exactly 30 seconds
    audio = whisper.pad_or_trim(audio)
    audio = audio.to(device)
    
    mel = whisper.log_mel_spectrogram(audio)
    
    with torch.no_grad():
        encoder_output = whisper_model.encoder(mel.unsqueeze(0))
    
    return encoder_output.squeeze(0)  # shape: (n_audio_ctx, n_audio_state)

def get_librispeech_datasets(whisper_model, adapter, tokenizer, device, train_size=500, val_size=100):
    """load and prepare LibriSpeech datasets"""
    # stream the dataset
    train_stream = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
        trust_remote_code=True
    )
    val_stream = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True
    )

    # cast audio column to in-memory array
    train_stream = train_stream.cast_column("audio", Audio(sampling_rate=16000))
    val_stream = val_stream.cast_column("audio", Audio(sampling_rate=16000))

    # define preprocessing function
    def preprocess_fn(batch):
        # extract Whisper features
        feats = extract_whisper_features_from_array(whisper_model, batch["audio"]["array"], device)
        feats = feats.cpu()
        
        # project through adapter
        with torch.no_grad():
            embeds = adapter(feats.unsqueeze(0)).squeeze(0)
        embeds = embeds.cpu()
        
        # tokenize text
        tok = tokenizer(batch["text"], truncation=True, max_length=512)
        
        return {
            "inputs_embeds": embeds,
            "attention_mask": torch.ones(embeds.size(0), dtype=torch.long),
            "labels": torch.tensor(tok["input_ids"], dtype=torch.long),
        }

    # map and slice datasets
    train_prepped = train_stream.map(preprocess_fn, remove_columns=["audio", "text"])
    small_train = train_prepped.take(train_size)
    val_prepped = val_stream.map(preprocess_fn, remove_columns=["audio", "text"])
    small_val = val_prepped.take(val_size)
    
    return small_train, small_val

# def collate_fn(batch):
#     """collate function to handle batches of examples"""
#     embed_list = [ex["inputs_embeds"] for ex in batch]
    
#     embeds_padded = pad_sequence(embed_list, batch_first=True)
    
#     attn_masks = []
#     for ex in batch:
#         mask = torch.cat([
#             ex["attention_mask"],
#             torch.zeros(embeds_padded.size(1) - ex["attention_mask"].size(0), dtype=torch.long)
#         ])
#         attn_masks.append(mask)
    
#     attn = torch.stack(attn_masks)
    
#     label_list = [ex["labels"] for ex in batch]
#     labels_padded = pad_sequence(label_list, batch_first=True, padding_value=-100)
    
#     return {
#         "inputs_embeds": embeds_padded,
#         "attention_mask": attn,
#         "labels": labels_padded,
#     }