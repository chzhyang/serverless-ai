
model_id = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english"

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
    save_dir = "/home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
    # Apply static quantization and save the resulting model in the OpenVINO IR format
    quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=save_dir)
    # Load the quantized model
    optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = classifier("He's a dreadful magician.")

static_int8()