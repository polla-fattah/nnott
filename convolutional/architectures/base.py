"""Shared helpers for convolutional architecture builders."""


class Architecture:
    name = "base"

    def build(self, num_classes=10, **kwargs):
        raise NotImplementedError

    def __call__(self, num_classes=10, **kwargs):
        return self.build(num_classes=num_classes, **kwargs)


__all__ = ["Architecture"]
