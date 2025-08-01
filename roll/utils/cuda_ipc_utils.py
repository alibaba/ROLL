"""
ref:https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils.py
"""
from multiprocessing.reduction import ForkingPickler
import pybase64
import io
class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """
        Serialize a Python object using ForkingPickler.

        Args:
            obj: The object to serialize.
            output_str (bool): If True, return a base64-encoded string instead of raw bytes.

        Returns:
            bytes or str: The serialized object.
        """
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """
        Deserialize a previously serialized object.

        Args:
            data (bytes or str): The serialized data, optionally base64-encoded.

        Returns:
            The deserialized Python object.
        """
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return ForkingPickler.loads(data)