import json
from torch.utils.data import IterableDataset

class HallucinationDataset(IterableDataset):
    def __init__(self, path, task):
        self.task = task
        with open(path, "r") as f:
            self.data = json.load(f)

    @staticmethod
    def _preprocess_elem(elem):
        new_elem = elem.copy()
        new_elem["label"] = int(elem["label"] == "Not Hallucination")
        return new_elem

    def __iter__(self):
        for elem in self.data:
            if elem["task"] == self.task:
                yield self._preprocess_elem(elem)
