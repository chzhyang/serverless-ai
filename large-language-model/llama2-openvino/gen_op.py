import openvino as ov
from transformers import LlamaTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import time
import argparse
import logging as log
import sys
log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-h',
                    '--help',
                    action='help',
                    help='Show this help message and exit.')
parser.add_argument('-m',
                    '--model_id',
                    required=True,
                    type=str,
                    help='Required. hugging face model id or local model path')
parser.add_argument('-p',
                    '--prompt',
                    default='What is AI?',
                    type=str,
                    help='Required. prompt sentence')
parser.add_argument('-l',
                    '--max_sequence_length',
                    default=128,
                    required=False,
                    type=int,
                    help='maximun lengh of output')
parser.add_argument('-d',
                    '--device',
                    default='CPU',
                    required=False,
                    type=str,
                    help='device for inference')
parser.add_argument('-a',
                    '--accuracy-mode',
                    default=False,
                    required=False,
                    type=bool,
                    help='')
parser.add_argument('--threads',
                    default=False,
                    required=False,
                    type=int,
                    help='')
args = parser.parse_args()

log.info(" --- load tokenizer --- ")
tokenizer = LlamaTokenizer.from_pretrained(
    args.model_id, trust_remote_code=True)

core = ov.Core()
# read config
def param_to_string(parameters) -> str:
    """Convert a list / tuple of parameters returned from IE to a string."""
    if isinstance(parameters, (list, tuple)):
        return ', '.join([str(x) for x in parameters])
    else:
        return str(parameters)
log.info('Available devices:')
for device in core.available_devices:
    log.info(f'{device}:')
    log.info('\tSUPPORTED_PROPERTIES:')
    for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
        if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
            try:
                property_val = core.get_property(device, property_key)
            except TypeError:
                property_val = 'UNSUPPORTED TYPE'
            log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
    log.info('')
# config
# config execution model hint(inference_precision)

if args.accuracy_mode:
    print(" --- set CPU execution hint --- ")
    core.set_property(
        "CPU",
        {ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.ACCURACY},
    )
# config ov
ov_config = {'PERFORMANCE_HINT': 'LATENCY',
             #  'NUM_STREAMS': '1',
             "CACHE_DIR": "./model_cache",
             }
if args.threads:
    ov_config["INFERENCE_NUM_THREADS"] = str(args.threads)
    log.info(f'ov_config: {ov_config}')


# load model
try:
    log.info(" --- use local model --- ")
    # model = OVModelForCausalLM.from_pretrained(
    #     args.model_id, compile=False, device=args.device)
    model = OVModelForCausalLM.from_pretrained(
        args.model_id, compile=False, device=args.device, ov_config=ov_config)
except:
    log.info(" --- use remote model --- ")
    model = OVModelForCausalLM.from_pretrained(
        args.model_id, compile=False, device=args.device, export=True)
model.compile()

inference_precision = core.get_property(
    "CPU", ov.properties.hint.inference_precision())
inference_num_threads = core.get_property(
    "CPU", ov.properties.inference_num_threads())
log.info(
    f'inference_precision: {inference_precision}, inference_num_threads: {inference_num_threads}')

# read config
log.info('Available devices:')
for device in core.available_devices:
    log.info(f'{device} :')
    log.info('\tSUPPORTED_PROPERTIES:')
    for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
        if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
            try:
                property_val = core.get_property(device, property_key)
            except TypeError:
                property_val = 'UNSUPPORTED TYPE'
            log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
    log.info('')

inputs = tokenizer(args.prompt, return_tensors="pt")
prompt_tokens = inputs.input_ids.shape[1]
perf = {"latency": []}

log.info(" --- start generating --- ")
st = time.perf_counter()
output_ids = model.generate(inputs.input_ids,
                            max_length=args.max_sequence_length+prompt_tokens,
                            perf=perf)
end = time.perf_counter()


completion_ids = output_ids[0].tolist()[prompt_tokens:]

log.info(" --- text decoding --- ")
completion = tokenizer.decode(completion_ids,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)

log.info(f"Generation took {end - st:.3f} s on {args.device}")
latency = perf["latency"]
print("latency len: ", len(latency))
result = {
    "completion": completion,
    "prompt_tokens": prompt_tokens,
    "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
    "completion_tokens": len(completion_ids),
    # total tokens completion latency, except tokenizer.decode time
    "total_token_latency_s": sum(latency),
    # first token completion latency
    "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
    # next token completion latency
    "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
    # average token completion latency
    "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
}
log.info(result)
