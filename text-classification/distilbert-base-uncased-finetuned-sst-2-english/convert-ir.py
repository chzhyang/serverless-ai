
model_id = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english"
ir_model = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-ir"

def convert_ir():
    from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline
    #  load a PyTorch checkpoint, set export=True to convert your model to the OpenVINO IR.
    model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(ir_model)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = classifier("He's a dreadful magician.")
    print(results)

convert_ir()