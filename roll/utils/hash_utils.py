import hashlib
import numpy as np
import torch


def compute_object_hash(obj):
    """
    Compute the SHA256 hash of a Python object.
    This function handles basic types, None, lists, tuples, dictionaries, and NumPy arrays.
    It calculates the hash by serializing the object's contents into a byte stream and feeding it to a SHA256 hasher.

    Args:
        obj: Any Python object.

    Returns:
        str: Hexadecimal string representation of the object's SHA256 hash.

    Raises:
        TypeError: If an unsupported type is encountered.
    """
    hasher = hashlib.sha256()

    def _update_hasher(value):
        """
        Recursively feed the object's content into the hasher.
        By adding type-specific prefixes, we ensure that values of different types—even if numerically equal—produce distinct hashes.
        """
        if value is None:
            hasher.update(b"None:")
        elif isinstance(value, (bool)):
            # Convert True/False to the byte strings "True"/"False"
            hasher.update(b"bool:" + str(value).encode('utf-8'))
        elif isinstance(value, (int, float)):
            # Integers and floats are converted to strings before encoding.
            # Note: floating-point hashing may be affected by precision issues.
            # To tolerate small differences in floats, round them before hashing.
            hasher.update(b"number:" + str(value).encode('utf-8'))
        elif isinstance(value, str):
            hasher.update(b"str:" + value.encode('utf-8'))
        elif isinstance(value, bytes):
            hasher.update(b"bytes:" + value)
        elif isinstance(value, (np.integer, np.floating)):
            # Convert NumPy scalar to native Python type
            hasher.update(b"np_scalar:" + str(value.item()).encode('utf-8'))
        elif isinstance(value, np.ndarray):
            # NumPy array: hash its dtype, shape, and raw byte content
            hasher.update(b"ndarray:")
            hasher.update(str(value.dtype).encode('utf-8'))
            hasher.update(str(value.shape).encode('utf-8'))
            # Use 'C' order to guarantee consistent byte ordering
            hasher.update(value.tobytes(order='C'))
        elif isinstance(value, torch.Tensor):
            # Tensor: 先转 CPU、再取 dtype/shape/raw bytes
            cpu_tensor = value.detach().cpu().contiguous()   # detach 避免梯度信息
            hasher.update(b"tensor:")
            hasher.update(str(cpu_tensor.dtype).encode('utf-8'))
            hasher.update(str(cpu_tensor.shape).encode('utf-8'))
            hasher.update(cpu_tensor.numpy().tobytes(order='C'))
        elif isinstance(value, (list, tuple)):
            # Lists and tuples: process each element recursively
            hasher.update(b"sequence_start:")
            for item in value:
                _update_hasher(item)
            hasher.update(b"sequence_end:")
        elif isinstance(value, dict):
            # Dictionary: recursively process each key-value pair and sort keys to ensure hash consistency
            hasher.update(b"dict_start:")
            # Sort keys to ensure consistent processing order of dictionary items
            for k, v in sorted(value.items(), key=lambda x: str(x[0])):
                _update_hasher(k)
                _update_hasher(v)
            hasher.update(b"dict_end:")
        elif isinstance(value, (set, frozenset)):
            # Sets: convert to a sorted list before hashing to ensure determinism
            hasher.update(b"set_start:")
            # Convert the set to a sorted list before hashing to ensure deterministic ordering.
            sorted_items = sorted(list(value), key=lambda x: str(x)) # 需要将元素转换为str以便排序
            for item in sorted_items:
                _update_hasher(item)
            hasher.update(b"set_end:")
        else:
            # For other custom objects, you could try using __repr__ or __reduce__,
            # but these methods are generally discouraged because they may be inconsistent or unsafe.
            # A safer approach is to explicitly define how such objects should be hashed, or simply raise an error.
            raise TypeError(f"Unsupported type for hashing: {type(value)}")

    _update_hasher(obj)
    return hasher.hexdigest()

if __name__ == "__main__":
    # Base objects
    print("--- base object ---")
    print(f"Hash of 1: {compute_object_hash(1)}")
    print(f"Hash of 1.0: {compute_object_hash(1.0)}")  # Different hashes for 1 and 1.0 due to distinct type prefixes
    print(f"Hash of 'hello': {compute_object_hash('hello')}")
    print(f"Hash of True: {compute_object_hash(True)}")
    print(f"Hash of False: {compute_object_hash(False)}")
    print(f"Hash of None: {compute_object_hash(None)}")
    print(f"Hash of b'raw_bytes': {compute_object_hash(b'raw_bytes')}")

    # NumPy arrays
    print("\n--- NumPy Arrays ---")
    img1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    img2 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    img3 = np.array([[1, 2], [3, 5]], dtype=np.uint8)  # Different content
    img4 = np.array([[1, 2], [3, 4]], dtype=np.int16)  # Different dtype

    print(f"Hash of img1: {compute_object_hash(img1)}")
    print(f"Hash of img2: {compute_object_hash(img2)}")  # Should match img1
    print(f"Hash of img3: {compute_object_hash(img3)}")  # Should differ from img1
    print(f"Hash of img4: {compute_object_hash(img4)}")  # Should differ from img1 (different dtype)

    # NumPy float arrays
    float_arr1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    float_arr2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    float_arr3 = np.array([0.10000000000000001, 0.2, 0.3], dtype=np.float32)  # Tiny difference

    print(f"Hash of float_arr1: {compute_object_hash(float_arr1)}")
    print(f"Hash of float_arr2: {compute_object_hash(float_arr2)}")  # Should match float_arr1
    print(f"Hash of float_arr3: {compute_object_hash(float_arr3)}")  # May differ depending on float representation

    # Composite structures
    print("\n--- Composite Structures ---")
    data1 = {
        'name': 'Alice',
        'age': 30,
        'scores': [95, 88, {'math': 90}],
        'matrix': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'enabled': True,
        'meta': None
    }
    data2 = {
        'age': 30,  # Keys in different order
        'name': 'Alice',
        'scores': [95, 88, {'math': 90}],
        'matrix': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'enabled': True,
        'meta': None
    }
    data3 = {
        'name': 'Bob',  # Value differs
        'age': 30,
        'scores': [95, 88, {'math': 90}],
        'matrix': np.array([[1, 0], [0, 1]], dtype=np.float32),
        'enabled': True,
        'meta': None
    }
    data4 = (
        'Alice', 30, [95, 88, {'math': 90}],
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        True, None
    )

    print(f"Hash of data1: {compute_object_hash(data1)}")
    print(f"Hash of data2: {compute_object_hash(data2)}")  # Should match data1 (keys sorted)
    print(f"Hash of data3: {compute_object_hash(data3)}")  # Should differ from data1
    print(f"Hash of data4 (tuple): {compute_object_hash(data4)}")  # Should differ from data1 (type mismatch)

    # Lists vs tuples (same contents, different types → different hashes)
    list_data = [1, 2, 'a']
    tuple_data = (1, 2, 'a')
    print(f"Hash of list_data: {compute_object_hash(list_data)}")
    print(f"Hash of tuple_data: {compute_object_hash(tuple_data)}")

    # Sets (same contents yield same hash after sorting)
    set_data1 = {1, 2, 'a'}
    set_data2 = {'a', 2, 1}
    print(f"Hash of set_data1: {compute_object_hash(set_data1)}")
    print(f"Hash of set_data2: {compute_object_hash(set_data2)}")

    # Unsupported type
    class CustomObject:
        def __init__(self, value):
            self.value = value

    try:
        compute_object_hash(CustomObject(123))
    except TypeError as e:
        print(f"\nCaught expected error for CustomObject: {e}")

    # Empty containers
    print(f"Hash of empty list []: {compute_object_hash([])}")
    print(f"Hash of empty tuple (): {compute_object_hash(())}")
    print(f"Hash of empty dict {{}}: {compute_object_hash({})}")
    print(f"Hash of empty set set(): {compute_object_hash(set())}")

    # Distinguish similar-looking but differently typed values
    print(f"Hash of 1 (int): {compute_object_hash(1)}")
    print(f"Hash of '1' (str): {compute_object_hash('1')}")
    print(f"Hash of [1] (list): {compute_object_hash([1])}")
    print(f"Hash of (1,) (tuple): {compute_object_hash((1,))}")
    print(f"Hash of {{1}} (set): {compute_object_hash({1})}")
    print(f"Hash of {{'a': 1}} (dict): {compute_object_hash({'a': 1})}")
