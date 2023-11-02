# from transformers import ViTImageProcessor, ViTForImageClassification
# from PIL import Image

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open("data/test1.jpg")

# # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# processor = ViTImageProcessor.from_pretrained('model/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('model/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

from transformers import AutoFeatureExtractor, pipeline
from optimum.intel.openvino import OVModelForImageClassification

model_id = "model/vit-base-patch16-224"
# Load a model from the HF hub and convert it to the OpenVINO format
model = OVModelForImageClassification.from_pretrained(model_id, from_transformers=True, export=True)
preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
onnx_path = "model/vit-base-patch16-224-ov-vnni"
# # quantization
# from optimum.onnxruntime import ORTQuantizer
# from optimum.onnxruntime.configuration import AutoQuantizationConfig
# # create ORTQuantizer and define quantization configuration
# dynamic_quantizer = ORTQuantizer.from_pretrained(model)
# dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# # apply the quantization configuration to the model
# model_quantized_path = dynamic_quantizer.quantize(
#     save_dir=onnx_path,
#     quantization_config=dqconfig,
# )

# import os
# # get model file size
# size = os.path.getsize(onnx_path / "model.onnx")/(1024*1024)
# quantized_model = os.path.getsize(onnx_path / "model_quantized.onnx")/(1024*1024)
# print(f"Model file size: {size:.2f} MB")
# print(f"Quantized Model file size: {quantized_model:.2f} MB")

model.save_pretrained(onnx_path)
preprocessor.save_pretrained(onnx_path)
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
# # 2GB memory
# cls_pipeline = pipeline("image-classification", model=save_directory, feature_extractor=feature_extractor)
# url = "data/cat.jpg"
# # Run inference with OpenVINO Runtime using Transformers pipelines
# outputs = cls_pipeline(url)
# max_item = max(outputs, key=lambda x: x['score'])
# max_label = max_item['label']
# print(max_label)