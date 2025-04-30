# minimal_adapter_test.py
import torch
import whisper
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model import WhisperAdapter
from src.data import extract_whisper_features_from_array
import librosa
import warnings

warnings.filterwarnings("ignore")

def main():
    """test for whisper-llama adapter"""
    parser = argparse.ArgumentParser(description="minimal adapter test")
    parser.add_argument("--audio", type=str, required=True, help="path to audio file")
    parser.add_argument("--adapter_path", type=str, default="adapter.pth", help="path to adapter checkpoint")
    
    args = parser.parse_args()
    
    # setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    # load models
    print("loading models...")
    whisper_model = whisper.load_model("tiny").to(device)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")
    llama = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B")
    llama.to(device)
    
    # load adapter
    whisper_dim = whisper_model.dims.n_audio_state  # 384
    llama_dim = llama.config.hidden_size  # 3072
    adapter = WhisperAdapter(whisper_dim, 1024, llama_dim).to(device)
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device, weights_only=True))
    
    # set all models to eval mode
    whisper_model.eval()
    llama.eval()
    adapter.eval()
    
    # get reference transcription
    print("processing audio...")
    audio_array, _ = librosa.load(args.audio, sr=16000, mono=True)
    reference = whisper_model.transcribe(audio_array)
    print(f"whisper transcript: {reference['text']}")
    
    # get adapter-based transcription
    with torch.no_grad():
        # extract whisper features
        feats = extract_whisper_features_from_array(whisper_model, audio_array, device)
        
        # pass through adapter
        embeds = adapter(feats.unsqueeze(0))
        
        # add prompt
        prompt = "Transcribe: "
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = llama.get_input_embeddings()(prompt_tokens.input_ids)
        
        # combine embeddings
        combined_embeds = torch.cat([prompt_embeds, embeds], dim=1)
        attention_mask = torch.ones(combined_embeds.size(0), combined_embeds.size(1), 
                                   device=device, dtype=torch.long)
        
        # generate text with conservative settings
        output_ids = llama.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            temperature=0.2,
            do_sample=True,
            num_beams=2,
            repetition_penalty=1.3
        )
    
    # decode the output (skip the prompt tokens)
    prompt_len = prompt_tokens.input_ids.size(1)
    output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    
    print(f"adapter transcription: {output_text}")
    
if __name__ == "__main__":
    main()