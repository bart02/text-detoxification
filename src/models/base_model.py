from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def generate(self, inputs):
        pass

    def _postprocessing(self, outputs):
        return outputs, []
