from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "./proj_vi_outputs"

#load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)

#load LoRA
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

#test the model
def get_answer(query):
    prompt = f"Instrustion:\n{query}\n\n Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded
    
if __name__ == "__main__":
    while True:
        question = input("enter question: \n")

        print(get_answer(question))