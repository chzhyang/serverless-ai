from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoFeatureExtractor
from pathlib import Path

# model_id="model/vit-base-patch16-224"
# onnx_path = Path("model/vit-base-patch16-224-onnx")
model_id="nateraw/vit-base-beans"
onnx_path = Path("model/vit-base-beans-onnx")

# load vanilla transformers and convert to onnx
model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
preprocessor = AutoFeatureExtractor.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
preprocessor.save_pretrained(onnx_path)