import torch


class Device:
    @staticmethod
    def get_available_device() -> torch.device:
        if torch.cuda.is_available():
            return Device.get_cuda()
        if torch.backends.mps.is_available():
            return Device.get_mps()
        return Device.get_cpu()

    @staticmethod
    def get_cpu():
        return torch.device('cpu')

    @staticmethod
    def get_mps():
        return torch.device('mps')

    @staticmethod
    def get_cuda():
        return torch.device('cuda')
