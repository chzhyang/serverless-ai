from transformers import DistilBertTokenizer, DistilBertModel
import torch
model_int8_path='/home/sdp/models/distilbert-base-cased-distilled-squad'
tokenizer = DistilBertTokenizer.from_pretrained(model_int8_path)
model = DistilBertModel.from_pretrained(model_int8_path)
print("model loaded")
import time
time.sleep(5)
print("inference")
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# predicted_class_id = outputs.logits.argmax().item()
# model.config.id2label[predicted_class_id]
print(outputs)