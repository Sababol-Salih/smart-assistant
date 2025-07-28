import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# تحميل النموذج
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# التاريخ الحواري
chat_history_ids = None

# دالة لإرسال الرسالة
def send_message():
    global chat_history_ids

    user_input = entry.get()
    chat_log.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    # ترميز الرسالة
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is None:
        chat_history_ids = new_input_ids
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # توليد الرد
    output_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # عرض الرد
    chat_log.insert(tk.END, f"Bot: {response}\n")

# إعداد نافذة الواجهة
window = tk.Tk()
window.title("Smart Assistant")

chat_log = tk.Text(window, height=20, width=60)
chat_log.pack()

entry = tk.Entry(window, width=50)
entry.pack(side=tk.LEFT, padx=5, pady=5)

send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack(side=tk.LEFT)

window.mainloop()
