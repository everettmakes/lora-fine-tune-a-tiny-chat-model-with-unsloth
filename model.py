"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length,
        load_in_4bit = True,
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum([p.numel() for p in model.parameters()])

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    return any(isinstance(m, Linear4bit) for m in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    if target_modules is None:
        target_modules = get_lora_target_modules()
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
    )
    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    return sum(p.numel() if p.requires_grad is True else 0 for p in model.parameters())

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a tiny hand-written instruction/response dataset for SFT."""
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "instruction": "Translate 'good morning' into Spanish.",
            "response": "'Good morning' in Spanish is 'buenos días'.",
        },
        {
            "instruction": "Write a Python function that returns the square of a number.",
            "response": "def square(x):\n    return x * x",
        },
        {
            "instruction": "Summarise in one sentence: LoRA trains small low-rank matrices instead of the full model weights.",
            "response": "LoRA fine-tunes a model cheaply by learning small add-on matrices while the original weights stay frozen.",
        },
        {
            "instruction": "Give me one tip for staying focused while studying.",
            "response": "Work in 25-minute blocks with your phone in another room, then take a 5-minute break.",
        },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(e) for e in examples]

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    return list(tokenizer(text).input_ids)

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return len(input_ids)

# Step 15 - build_training_arguments
def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    params = {
        'per_device_train_batch_size': 1, 
        'max_steps': max_steps,
        'learning_rate': learning_rate,
        'output_dir': output_dir,
        'logging_steps': 1,
        'optim': 'adamw_8bit'
    }
    if torch.cuda.is_bf16_supported():
        params['bf16'] = True
    else:
        params['fp16'] = True
    return TrainingArguments(**params)

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

