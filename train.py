import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from src.model import WhisperAdapter
from src.data import get_librispeech_datasets
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_loss", float('inf'))
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
                
        if self.counter >= self.patience:
            control.should_training_stop = True
            print(f"Early stopping triggered after {state.global_step} steps!")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    whisper_model = whisper.load_model("tiny").to(device)
    whisper_model.eval()
    for p in whisper_model.parameters():
        p.requires_grad = False
    
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

        # stack & embed token IDs
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
        whisper_model, adapter, tokenizer, device, train_size=1000, val_size=100  # Reduce dataset size
    )
    
    # define training arguments
    print("setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="llama_whisper_adapter",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-6,
        weight_decay=0.01,
        fp16=(device=="cuda"),
        gradient_accumulation_steps=4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,  
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        max_steps=2000,
        report_to="none",
        dataloader_pin_memory=False
    )
    
    # create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=llama,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  
        callbacks=[EarlyStoppingCallback(patience=3)]
    )
    
    # train
    print("starting training...")
    trainer.train()
    print("training complete!")
    
    # save adapter
    torch.save(adapter.state_dict(), "adapter.pth")
    print("adapter saved to adapter.pth")

if __name__ == "__main__":
    main()