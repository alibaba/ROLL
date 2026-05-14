from .unknown import UnknownPlatform


class CorexPlatform(UnknownPlatform):
    device_name: str = "COREX"

    @classmethod
    def is_cuda(cls) -> bool:
        return True
