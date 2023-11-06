from transformers import pipeline
import torch
model_path='/home/sdp/models/distilbert-base-cased-distilled-squad'
question_answerer = pipeline("question-answering", model=model_path)
context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""
q1="What is a good example of a question answering dataset?"
context2 = r"""
Alice is sitting on the bench. Bob is sitting next to her.
"""
q2="Who is the CEO?"
result = question_answerer(question=q2,
                           context=context2)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")