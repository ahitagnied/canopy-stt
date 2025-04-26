import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from src.model import WhisperAdapter
from src.data import get_librispeech_datasets
from torch.nn.utils.rnn import pad_sequence

def main():
    # use GPU for LLaMA, but CPU for Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    print(f"using device for LLaMA: {device}")
    print(f"using device for Whisper: {device}")
    
    # load Whisper model on CPU only
    print("loading Whisper model on CPU...")
    whisper_model = whisper.load_model("tiny", device="cpu")
    whisper_model.eval()
    for p in whisper_model.parameters():
        p.requires_grad = False
    
    # load llama model and tokenizer on GPU
    print("loading llama model and tokenizer...")
    MODEL_ID = "unsloth/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llama = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    llama.to(device)
    
    def collate_fn(batch):
        # stack audio‐prefix embeddings
        prefix_list   = [ex["inputs_embeds"] for ex in batch]                   # each is (A, D)
        prefix_embeds = pad_sequence(prefix_list, batch_first=True).to(device)  # (B, A, D)

        B, A, D = prefix_embeds.size()

        # stack & embed your token IDs
        ids_list   = [ex["labels"] for ex in batch]                             # each is (T_i,)
        input_ids  = pad_sequence(ids_list, batch_first=True,
                    padding_value=tokenizer.pad_token_id).to(device)            # (B, T)
        T = input_ids.size(1)
        token_embeds = llama.get_input_embeddings()(input_ids)                  # (B, T, D)

        # concat prefix + token embeddings
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)         # (B, A+T, D)

        # build labels so that the first A positions are ignored
        prefix_labels = torch.full((B, A), -100, dtype=torch.long, device=device)
        labels        = torch.cat([prefix_labels, input_ids], dim=1)            # (B, A+T)

        # full attention mask
        attention_mask = torch.ones((B, A + T), dtype=torch.long, device=device)

        return {
            "inputs_embeds":  inputs_embeds,
            "attention_mask": attention_mask,
            "labels":          labels,
        }

    # define model dimensions and initialize adapter
    WHISPER_DIM = whisper_model.dims.n_audio_state  # 512
    LLAMA_DIM = llama.config.hidden_size            # 3072
    ADAPTER_HID = 1024
    
    print(f"WHISPER_DIM: {WHISPER_DIM}, LLAMA_DIM: {LLAMA_DIM}")
    
    adapter = WhisperAdapter(WHISPER_DIM, ADAPTER_HID, LLAMA_DIM).to(device)
    
    # load datasets
    print("Loading and preprocessing datasets...")
    train_dataset, val_dataset = get_librispeech_datasets(
        whisper_model, adapter, tokenizer, device, train_size=50, val_size=10  # Reduce dataset size
    )
    
    # define training arguments
    print("setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="llama_whisper_adapter",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        fp16=(device=="cuda"),
        logging_steps=20,
        save_strategy="epoch",
        max_steps=500,
        report_to="none",
    )
    
    # create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=llama,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  
    )
    
    # train
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    # save adapter
    torch.save(adapter.state_dict(), "whisper_llama_adapter.pth")
    print("adapter saved to whisper_llama_adapter.pth")

if __name__ == "__main__":
    main()