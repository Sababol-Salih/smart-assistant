from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/GPT-2")  # أبسط من DialoGPT
model = AutoModelForCausalLM.from_pretrained("microsoft/GPT-2")

print("🤖 Smart Assistant is ready. Type 'exit' to quit.")

chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break

    prompt = chat_history + f"You: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Bot:")[-1].strip()
    print("Bot:", response)

    chat_history += f"You: {user_input}\nBot: {response}\n"
