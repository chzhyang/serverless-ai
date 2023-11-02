from transformers import pipeline, AutoFeatureExtractor
from optimum.intel.openvino import OVModelForImageClassification

onnx_path="model/vit-base-patch16-224-ov"
ov_model = OVModelForImageClassification.from_pretrained(onnx_path)
preprocessor = AutoFeatureExtractor.from_pretrained(onnx_path)
ov_pipe = pipeline("image-classification", model=ov_model, feature_extractor=preprocessor)
outputs = ov_pipe("data/cat.jpg")
print(outputs)