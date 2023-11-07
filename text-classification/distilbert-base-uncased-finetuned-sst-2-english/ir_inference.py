from optimum.intel import OVModelForSequenceClassification
from transformers import DistilBertTokenizer, pipeline
import gc
model_ir_int8_static = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-int8-static" # 1GB memory
model_ir_fp32 = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-ir"
def ir_pipeline_inference(ir_model):
    ov_model = OVModelForSequenceClassification.from_pretrained(ir_model)
    tokenizer = DistilBertTokenizer.from_pretrained(ir_model)
    print("loaded model, sleep 5s")
    import time
    time.sleep(5)
    # warm up
    classifier = pipeline("text-classification", model=ov_model, tokenizer=tokenizer)
    classifier("I really like the new design of your website!")
    results=""
    # bench
    latency = []
    steps = 20
    st = time.perf_counter()
    for i in range(steps):
        results = classifier("I really like the new design of your website!")
    end = time.perf_counter()
    print(results)
    print("latency:", (end-st)*1000/steps)
    del ov_model
    gc.collect()

# ir_pipeline_inference(ir_model=model_ir_int8_static)
ir_pipeline_inference(ir_model=model_ir_fp32)