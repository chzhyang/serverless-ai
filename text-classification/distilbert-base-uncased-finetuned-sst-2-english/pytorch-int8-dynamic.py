from optimum.intel import INCModelForSequenceClassification
from transformers import pipeline, AutoTokenizer
import torch
model_int8_dynamic_oneline="Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model_int8_dynamic_local = "/home/sdp/models/intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
print("load model")
model = INCModelForSequenceClassification.from_pretrained(model_int8_dynamic_local)
text = "I really like the new design of your website!"
tokenizer = AutoTokenizer.from_pretrained(model_int8_dynamic_local)
print("inference")
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
output=model.config.id2label[predicted_class_id]
print(output)
