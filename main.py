import torch
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from src.model import WhisperAdapter
from src.data import get_librispeech_datasets, collate_fn

def main():
    # use GPU for LLaMA, but CPU for Whisper
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_device = "cpu"
    
    print(f"Using device for LLaMA: {gpu_device}")
    print(f"Using device for Whisper: {cpu_device}")
    
    # load Whisper model on CPU only
    print("Loading Whisper model on CPU...")
    whisper_model = whisper.load_model("tiny").to(cpu_device)
    whisper_model.eval()
    for p in whisper_model.parameters():
        p.requires_grad = False
    
    # Load LLaMA model and tokenizer on GPU
    print("Loading LLaMA model and tokenizer...")
    MODEL_ID = "unsloth/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llama = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    llama.to(gpu_device)
    
    # 3. Define model dimensions and initialize adapter
    WHISPER_DIM = whisper_model.dims.n_audio_state  # 512
    LLAMA_DIM = llama.config.hidden_size            # 3072
    ADAPTER_HID = 1024
    
    print(f"WHISPER_DIM: {WHISPER_DIM}, LLAMA_DIM: {LLAMA_DIM}")
    
    adapter = WhisperAdapter(WHISPER_DIM, ADAPTER_HID, LLAMA_DIM).to(device)
    
    # load datasets
    print("Loading and preprocessing datasets...")
    train_dataset, val_dataset = get_librispeech_datasets(
        whisper_model, adapter, tokenizer, cpu_device, train_size=50, val_size=10  # Reduce dataset size
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