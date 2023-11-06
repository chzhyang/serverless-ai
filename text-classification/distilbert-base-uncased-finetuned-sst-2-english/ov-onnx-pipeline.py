from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer
model_int8_path = "/home/sdp/models/intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model = ORTModelForSequenceClassification.from_pretrained(model_int8_path)
tokenizer = AutoTokenizer.from_pretrained(model_int8_path)
sa=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "I really like the new design of your website!"
output=sa(text)
print(output)