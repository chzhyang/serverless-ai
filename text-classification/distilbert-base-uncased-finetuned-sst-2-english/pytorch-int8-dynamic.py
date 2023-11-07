from optimum.intel import INCModelForSequenceClassification
from transformers import DistilBertTokenizer
import torch
import gc
model_int8_dynamic_oneline="Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model_int8_dynamic = "/home/sdp/models/intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"

text = "I really like the new design of your website!"

def torch_inference_int8_dynamic(model_inc):
    print("--- torch_inference_int8_dynamic ---")
    model = INCModelForSequenceClassification.from_pretrained(model_inc)
    tokenizer = DistilBertTokenizer.from_pretrained(model_inc)
    print("loaded model, sleep 5s")
    import time
    time.sleep(5)

    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        print("--- warm up ---")
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()

        print("--- bench ---")
        steps = 20
        st = time.perf_counter()
        for i in range(steps):
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            output=model.config.id2label[predicted_class_id]
        end = time.perf_counter()

    print("output:",output)
    print("latency:", (end-st)*1000/steps)
    
    del model
    gc.collect()

def pipeline_int8_dynamic(model_inc):
    print("--- pipeline_int8_dynamic ---")
    from transformers import pipeline
    model = INCModelForSequenceClassification.from_pretrained(model_inc)
    tokenizer = DistilBertTokenizer.from_pretrained(model_inc)
    sa=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("--- loaded model, sleep 5s ---")
    import time
    time.sleep(5)
    
    print("--- warm up ---")
    output=sa(text)

    # bench
    print("--- bench ---")
    steps = 20
    st = time.perf_counter()
    for i in range(steps):
        output = sa(text)
    end = time.perf_counter()
    print("output:",output)
    print("latency:", (end-st)*1000/steps)
    del model
    gc.collect()

torch_inference_int8_dynamic(model_int8_dynamic)
pipeline_int8_dynamic(model_int8_dynamic)