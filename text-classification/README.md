INT8	FP32
Accuracy (eval-accuracy)	0.9025	0.9106
Model size (MB)	165	256

FP32 https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english

INT8 https://huggingface.co/Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic

https://github.com/huggingface/optimum-intel/tree/main

python -m pip install "optimum-intel[extras]"@git+https://github.com/huggingface/optimum-intel.git
where extras can be one or more of neural-compressor, openvino, nncf.

Dynamic quantization can be used through the Optimum command-line interface based on Neural Compressor: https://github.com/huggingface/optimum-intel/tree/main#neural-compressor
Note that quantization is currently only supported for CPUs (only CPU backends are available)

Static quantization using NCCF: https://github.com/huggingface/optimum-intel/tree/main#post-training-static-quantization