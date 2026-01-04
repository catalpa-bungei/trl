
import json
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor

# 1. Load the dataset
train_dataset = load_dataset("json", data_files="/mnt/shared-storage-user/yangxuqing/post_processing/create_training_data/data/text/text_filtered_combined-shuffled-train.jsonl", split="train")
evaluation_dataset = load_dataset("json", data_files="/mnt/shared-storage-user/yangxuqing/post_processing/create_training_data/data/text/text_filtered_combined-shuffled-test.jsonl", split="test")

# 2. Convert to conversational language modeling data
def preprocess_function(example):
    # The user wants to convert to conversational format.
    # Based on the file content:
    # text_prompt -> user message
    # model_output -> assistant message
    
    messages = [
        {"role": "user", "content": example["text_prompt"]},
        {"role": "assistant", "content": f"<ANSWER is: \\boxed{{{example['ground_truth']}}}>|<CONFIDENCE:{int(example['avg_accuracy']*10)}>"}
    ]
    
    # If there was an image, we would handle it here, but the sampled data showed null images.
    # Qwen2.5-VL expects specific formatting for images if they exist.
    # For now, we proceed with text-only messages.
    # We must provide 'images' key because DataCollatorForLanguageModeling expects it in this version of TRL
    return {"messages": messages, "images": []}

# Apply the transformation
train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
evaluation_dataset = evaluation_dataset.map(preprocess_function, remove_columns=evaluation_dataset.column_names)


# 3. Setup for SFT
# User mentioned Qwen2.5VL-7B. Assuming the Hugging Face model ID.
model_id = "/mnt/shared-storage-user/yangxuqing/models/Qwen2.5VL-7B-Instruct/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4" 

# Configure SFT
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_length=2048, # Adjust as needed
    packing=False, # Typically False for conversational data unless handled carefully
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=10,
    num_train_epochs=1,
    dataset_text_field="messages", # SFTTrainer will automatically handle 'messages' column
    gradient_checkpointing=True, # Enable gradient checkpointing to save memory
    # bf16=True, # Enable bf16 mixed precision
    eval_strategy="steps",
    eval_steps=10,
    # save_strategy="steps",
    # save_steps=30,
    # load_best_model_at_end=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Initialize Trainer
# Note: For Qwen2.5-VL, we might need to pass the processor or specific formatting.
# SFTTrainer suppotrain_dataset,
processor = AutoProcessor.from_pretrained(model_id)
trainer = SFTTrainer(
    model=model_id,
    train_dataset=train_dataset,
    args=sft_config,
    eval_dataset=evaluation_dataset,
    processing_class=processor,
)

# 4. Train
trainer.train() 
# Commented out to avoid actual training execution during this interaction, 
# but the file is ready for the user to run.
