import torch


class AvailableDevice:
    @staticmethod
    def get() -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
