from optimum.intel import OVModelForSequenceClassification
from transformers import pipeline, DistilBertTokenizer
model_int8_path = "/home/sdp/models/intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model = OVModelForSequenceClassification.from_pretrained(model_int8_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_int8_path)
sa=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "I really like the new design of your website!"
output=sa(text)
print(output)