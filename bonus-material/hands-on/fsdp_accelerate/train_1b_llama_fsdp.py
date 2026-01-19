import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="FSDP training script for Causal LM")
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-3.2-1B", 
        help="Hugging Face model ID to train."
    )
    # Note: Llama 3.2 models require an auth token. 
    # It's best to log in via `huggingface-cli login` first.
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="wikimedia/wikipedia", 
        help="Dataset name."
    )
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="20231101.nl", 
        help="Dataset config/subset (e.g., language code)."
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Llama-1B-wikipedia-nl-fsdp", 
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Per-device training batch size."
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=1, 
        help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate."
    )
    parser.add_argument(
        "--seq_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length."
    )
    
    return parser.parse_args()

def load_streaming_dataset(dataset_name, dataset_config):
    """Loads the specified streaming dataset."""
    print(f"Loading streaming dataset: {dataset_name} (config: {dataset_config})")
    try:
        train_dataset = load_dataset(
            dataset_name, 
            dataset_config, 
            split="train[:2%]", 
            #split="train",
            streaming=stream,

        )
        return train_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model_and_tokenizer(model_name):
    """Loads the model and tokenizer, setting the pad token."""
    print(f"Loading model and tokenizer: {model_name}")
    
    # You must be logged in via `huggingface-cli login` to load meta-llama models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 trust_remote_code=True, 
                                                 #attn_implementation="flash_attention_2",
                                                 dtype=torch.bfloat16)

    # Set pad token to EOS token if it's not already set
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def main():
    """Main training function."""
    args = parse_args()
    
    # 1. Load Dataset
    train_dataset = load_streaming_dataset(args.dataset_name, args.dataset_config)
    if train_dataset is None:
        return

    # 2. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 3. Configure FSDP Training
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text", 
        max_length=args.seq_length,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_pin_memory=True,
        packing=True,
        max_steps=95367,
        gradient_checkpointing=False,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        report_to="tensorboard", 
        use_liger_kernel=False, #True,
        
        # FSDP settings
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_activation_checkpointing": "FULL"
        },
    )

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
    )

    # 5. Run Training
    print("Starting FSDP training...")
    trainer.train()
    
    print("Training complete. Saving model...")
    trainer.save_model()
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
