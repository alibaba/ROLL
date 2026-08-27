from collections.abc import Mapping

from roll.utils.fp8 import is_mxfp8_ascend


class QuantDescriptionMapping(Mapping):
    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class ConfigLike:
    def __init__(self, quant_description):
        self.quant_description = quant_description


def test_is_mxfp8_ascend_detects_top_level_mapping():
    assert is_mxfp8_ascend({"quant_method": "ascend"})


def test_is_mxfp8_ascend_detects_mapping_quant_description():
    quant_description = QuantDescriptionMapping({"quant_method": "ascend"})

    assert is_mxfp8_ascend(ConfigLike(quant_description))


def test_is_mxfp8_ascend_rejects_non_ascend_configs():
    assert not is_mxfp8_ascend({"quant_method": "fp8"})
    assert not is_mxfp8_ascend(ConfigLike({"quant_method": "fp8"}))
