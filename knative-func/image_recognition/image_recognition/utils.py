import json
import numpy as np

LABELS_FILE = "data/labellist.json"

def get_top_predictions(results, ids_are_one_indexed=False, preds_to_print=5):
  """Given an array of mode, graph_name, predicted_ID, print labels."""
  labels = get_labels()

  print("Predictions:")
  predictions = []
  for result in results:
    pred_ids = top_predictions(result, preds_to_print)
    pred_labels = get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
    predictions.append(pred_labels)
  return predictions

def get_labels():
  """Get the set of possible labels for classification."""
  with open(LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels

def top_predictions(result, n):
  """Get the top n predictions given the array of softmax results."""
  # We only care about the first example.
  probabilities = result
  # Get the ids of most probable labels. Reverse order to get greatest first.
  ids = np.argsort(probabilities)[::-1]
  
  return ids[:n]

def get_labels_for_ids(labels, ids, ids_are_one_indexed=False):
  """Get the human-readable labels for given ids.
  Args:
    labels: dict, string-ID to label mapping from ImageNet.
    ids: list of ints, IDs to return labels for.
    ids_are_one_indexed: whether to increment passed IDs by 1 to account for
      the background category. See ArgParser `--ids_are_one_indexed`
      for details.
  Returns:
    list of category labels
  """
  for x in ids:
    print(x, "=", labels[str(x + int(ids_are_one_indexed))])
  return [labels[str(x + int(ids_are_one_indexed))] for x in ids]