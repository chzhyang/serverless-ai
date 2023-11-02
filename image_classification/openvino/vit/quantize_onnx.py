from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import pipeline, AutoFeatureExtractor
# onnx_path="model/vit-base-patch16-224-onnx"
onnx_path="model/vit-base-beans-onnx"
# https://www.philschmid.de/optimizing-vision-transformer

# ov_model = OVModelForImageClassification.from_pretrained(onnx_path)
# preprocessor = AutoFeatureExtractor.from_pretrained(onnx_path)
# create ORTQuantizer and define quantization configuration
dynamic_quantizer = ORTQuantizer.from_pretrained(onnx_path)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
# quantize_path = "model/vit-base-patch16-224-onnx-quantize"
quantize_path = "model/vit-base-beans-onnx-quantize"
# apply the quantization configuration to the model
model_quantized_path = dynamic_quantizer.quantize(
    save_dir=quantize_path,
    quantization_config=dqconfig,
)

# import os

# # get model file size
# size = os.path.getsize(onnx_path+"/model.onnx")/(1024*1024)
# quantized_model = os.path.getsize(quantize_path+"/model_quantized.onnx")/(1024*1024)

# print(f"Model file size: {size:.2f} MB")
# print(f"Quantized Model file size: {quantized_model:.2f} MB")

from optimum.onnxruntime import ORTModelForImageClassification
from transformers import pipeline, AutoFeatureExtractor

model = ORTModelForImageClassification.from_pretrained(quantize_path, file_name="model_quantized.onnx")
preprocessor = AutoFeatureExtractor.from_pretrained(quantize_path)

q8_clf = pipeline("image-classification", model=model, feature_extractor=preprocessor)