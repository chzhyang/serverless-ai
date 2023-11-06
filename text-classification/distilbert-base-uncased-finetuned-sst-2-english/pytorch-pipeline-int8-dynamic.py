from optimum.intel import INCModelForSequenceClassification
from transformers import pipeline, DistilBertTokenizer
model_int8_dynamic_oneline="Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model_int8_dynamic_local = "/home/sdp/models/intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
print("load model")
model = INCModelForSequenceClassification.from_pretrained(model_int8_dynamic_local)
tokenizer = DistilBertTokenizer.from_pretrained(model_int8_dynamic_local)
print("inference")
sa=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "I really like the new design of your website!"
output=sa(text)
print(output)
