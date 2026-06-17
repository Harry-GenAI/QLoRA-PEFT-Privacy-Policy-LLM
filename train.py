from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model

#dataset
dataset = load_dataset("json", data_files="dataset.jsonl")

#model name
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

#prompt template
prompt_template ="""
Instruction:
{}

Response:
{}
"""

#preprocessing fun
def format_fun(example):
    prompt = f'Instruction:\n{example["instruction"]}\n\n'
    response = example["response"]

    full_text = prompt + response + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    labels =tokenized["input_ids"].copy()

    #!mask the prompt only train on response
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized

#tokenized dataset
tokenized_dataset = dataset.map(format_fun)

#print(dataset, "\n\n" , trained_dataset)

#QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype = torch.float16
)


#model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

#lora config - how model shud learn
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)




#wrap the model with LoRA
model = get_peft_model(model, lora_config)

#trainig args - how training runs
training_args = TrainingArguments(
    output_dir = "./proj_vi_outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    fp16=True,
    save_strategy="no"
)

#trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=training_args
)

trainer.train()

#save
model.save_pretrained("./proj_vi_outputs")
tokenizer.save_pretrained("./proj_vi_outputs")




