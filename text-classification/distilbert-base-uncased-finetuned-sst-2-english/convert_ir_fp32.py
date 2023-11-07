from optimum.intel import OVModelForSequenceClassification
from transformers import DistilBertTokenizer, pipeline
import gc
model_id = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english"
ir_model = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-ir"

def convert_ir():
    #  load a PyTorch checkpoint, set export=True to convert your model to the OpenVINO IR.
    model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = DistilBertTokenizer.from_pretrained(model_id)
    model.save_pretrained(ir_model)
    tokenizer.save_pretrained(ir_model)
    # classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    # results = classifier("He's a dreadful magician.")
    # print(results)
    del model
    gc.collect()

def ir_inference():
    ov_model = OVModelForSequenceClassification.from_pretrained(ir_model)
    tokenizer = DistilBertTokenizer.from_pretrained(ir_model)
    print("loaded model, sleep 5s") # 1.8GB memory - ice lake
    import time
    time.sleep(5)
    classifier = pipeline("text-classification", model=ov_model, tokenizer=tokenizer)
    results = classifier("He's a dreadful magician.")
    print(results)
    del ov_model
    gc.collect()

convert_ir()
# ir_inference()