"""Precision kwarg mapping: Ultralytics `half` -> `quantize` deprecation."""

import sys
import types

from src.detector import _precision_predict_kwargs


def _fake_cfg(monkeypatch, keys):
    mod = types.ModuleType("ultralytics.cfg")
    mod.DEFAULT_CFG_DICT = {k: None for k in keys}
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", mod)


def test_uses_quantize_when_supported(monkeypatch):
    _fake_cfg(monkeypatch, ["quantize", "conf"])
    assert _precision_predict_kwargs(True) == {"quantize": 16}
    assert _precision_predict_kwargs(False) == {"quantize": None}


def test_falls_back_to_half_on_older_ultralytics(monkeypatch):
    _fake_cfg(monkeypatch, ["half", "conf"])
    assert _precision_predict_kwargs(True) == {"half": True}
    assert _precision_predict_kwargs(False) == {"half": False}
