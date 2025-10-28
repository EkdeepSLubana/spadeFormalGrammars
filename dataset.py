import json
from typing import Any
from dataclasses import dataclass
# from nnsight.envoy import Envoy

values_ordered = ["tense-present", "tense-past",
                  "voice-active", "voice-passive",
                  "domain-science", "domain-fantasy", "domain-news", "domain-other",
                  "reading-level-high", "reading-level-low",
                  "sentiment-positive", "sentiment-neutral", "sentiment-negative"]

class Dataset:
    def __init__(self, location="data/labeled_sentences.jsonl"):
        self.location = location
        self.examples = []
        self.labels = {}
        self.labels_binary = {}
        self.load_data()

    def load_data(self):
        # Load examples and labels
        with open(self.location, 'r') as lines:
            for line in lines:
                data = json.loads(line)
                self.examples.append(data["sentence"])
                for key in data.keys():
                    if key == "sentence":
                        continue
                    if key not in self.labels:
                        self.labels[key] = []
                    self.labels[key].append(data[key])
        
        # Construct binarized version of labels for all key/value pairs
        for value in values_ordered:
            if value not in self.labels_binary:
                self.labels_binary[value] = []
        for key in self.labels:
            if key == "reading_level":
                is_True = [s > 11.5 for s in self.labels[key]]
                self.labels_binary["reading-level-high"] = is_True
                self.labels_binary["reading-level-low"] = [(s is False) for s in is_True]
                continue
            values = set(self.labels[key])
            for value in values:
                kv = f"{key}-{value}"
                # if kv not in self.labels_binary:
                #     self.labels_binary[kv] = []
                is_kv = [v == value for v in self.labels[key]]
                self.labels_binary[kv] = is_kv

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Any
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        if self.use_input:
            out = self.submodule.input # TODO make sure I didn't break for pythia
        else:
            out = self.submodule.output
        if self.is_tuple:
            return out[0]
        else:
            return out

    def set_activation(self, x):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0][:] = x
            else:
                self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output[:] = x

    def stop_grad(self):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0].grad = t.zeros_like(self.submodule.input[0])
            else:
                self.submodule.input.grad = t.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = t.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = t.zeros_like(self.submodule.output)