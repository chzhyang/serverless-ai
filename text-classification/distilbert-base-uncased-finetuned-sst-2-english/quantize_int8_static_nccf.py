import gc
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
q8_model = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
def static_int8():
    # apply static quantization on a fine-tuned DistilBERT
    # nccf
    from functools import partial
    from optimum.intel import OVQuantizer, OVModelForSequenceClassification
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    def preprocess_fn(examples, tokenizer):
        return tokenizer(
            examples["sentence"], padding=True, truncation=True, max_length=128
        )
    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        "glue",
        dataset_config_name="sst2",
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
        num_samples=100,
        dataset_split="train",
        preprocess_batch=True,
    )
    # The directory where the quantized model will be saved
    save_dir = q8_model
    # Apply static quantization and save the resulting model in the OpenVINO IR format
    quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=save_dir)
    tokenizer.save_pretrained(save_dir)
    gc.collect()
    
static_int8()