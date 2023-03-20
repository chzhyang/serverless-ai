python3 /home/models/image_recognition/tensorflow/resnet50/inference/eval_image_classifier_inference.py --input-graph=/home/resnet50_fp32_pretrained_model.pb --num-inter-threads=1  --num-intra-threads=56  --num-cores=56  --batch-size=1 --warmup-steps=1 --steps=100 --data-num-inter-threads=32 --data-num-intra-thread=14 --model-optimize=true
readmodel:   0.300014 sec
optimize:    55.627771 sec
loadmodel:   0.404054 sec
total model: 56.331912 sec

Average time: 0.010149 sec