from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import torch.nn.functional as F
import os

app = Flask(__name__)

# Load model and tokenizer
model_path = "lora_bert"  # update this if needed

tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

label_list = ["non_hate_speech", "hate_speech", "neither"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

    return {
        "text": text,
        "predicted_label": label_list[predicted_class],
        "probabilities": {label_list[i]: round(p.item(), 3) for i, p in enumerate(probs[0])}
    }

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        input_text = request.form.get("input_text")
        if input_text:
            result = predict(input_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
