"""Stopping tests without importing unrelated neural-network dependencies."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def thin(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for name in ('utils', 'pid'):
        package = types.ModuleType(name)
        package.__path__ = [str(root / name)]
        monkeypatch.setitem(sys.modules, name, package)

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, root.joinpath(*name.split('.')).with_suffix('.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    sys.modules['utils'].pinv = load('utils.linalg').pinv
    sys.modules['utils'].whiten = load('utils.estimate_channel').whiten
    load('pid.tilde_pid')
    return load('pid.thin_pid')


@pytest.fixture
def channels():
    return np.array([[0.2]]), np.array([[0.7]])


def test_target_accepts_initial_objective_without_update(thin, channels, monkeypatch):
    monkeypatch.setattr(thin, 'objective', lambda *args: 2.0)
    def unexpected(*args):
        pytest.fail('target reached: no gradient step should run')
    monkeypatch.setattr(thin, 'gradient', unexpected)
    _, obj, iterations, _ = thin.exact_thin_pid_minimizer(
        *channels, objective_target=2.0, ret_obj=True)
    assert obj == 2.0
    assert iterations == 1


def test_target_after_updates_and_swap(thin, monkeypatch):
    values = iter([3.0, 2.0, 1.0])
    monkeypatch.setattr(thin, 'objective', lambda *args: next(values))
    sig, obj, iterations, _ = thin.exact_thin_pid_minimizer(
        np.array([[0.2]]), np.array([[0.7], [0.3]]),
        objective_target=1.5, native_stopping=False, ret_obj=True)
    assert sig.shape == (1, 2)
    assert obj == 1.0
    assert iterations == 3


def test_native_stopping_can_be_disabled(thin, channels, monkeypatch):
    def run(native):
        values = iter([2.0] * 25 + [1.0])
        monkeypatch.setattr(thin, 'objective', lambda *args: next(values))
        return thin.exact_thin_pid_minimizer(
            *channels, ret_obj=True, objective_target=1.0,
            native_stopping=native)
    assert run(True)[1:3] == (2.0, 21)
    assert run(False)[1:3] == (1.0, 26)


@pytest.mark.parametrize('phase', ['initialization', 'objective', 'gradient', 'projection'])
def test_timeout_between_operations(thin, channels, monkeypatch, phase):
    clock = [0.0]
    monkeypatch.setattr(thin.time, 'monotonic', lambda: clock[0])
    function = {'initialization': 'pinv', 'objective': 'objective',
                'gradient': 'gradient', 'projection': 'thin_project'}[phase]
    original = getattr(thin, function)
    calls = [0]
    def expire(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if phase != 'projection' or calls[0] == 2:
            clock[0] = 1.0
        return result
    monkeypatch.setattr(thin, function, expire)
    with pytest.raises(TimeoutError, match='Thin-PID'):
        thin.exact_thin_pid_minimizer(*channels, timeout=1.0, native_stopping=False)


@pytest.mark.parametrize('kwargs', [
    {'timeout': 0}, {'timeout': -1}, {'timeout': np.nan},
    {'timeout': np.inf}, {'timeout': [1]},
    {'objective_target': np.nan}, {'objective_target': np.inf},
    {'objective_target': [1]},
])
def test_invalid_stopping_options(thin, channels, kwargs):
    with pytest.raises(ValueError):
        thin.exact_thin_pid_minimizer(*channels, **kwargs)


def test_default_options_preserve_results(thin, channels):
    default = thin.exact_thin_pid_minimizer(*channels, ret_obj=True)
    explicit = thin.exact_thin_pid_minimizer(
        *channels, ret_obj=True, objective_target=None,
        native_stopping=True, timeout=None)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)


def test_covariance_wrapper_forwards_options(thin, monkeypatch):
    received = {}
    def minimizer(*args, **kwargs):
        received.update(kwargs)
        raise TimeoutError('test deadline')
    monkeypatch.setattr(thin, 'exact_thin_pid_minimizer', minimizer)
    h = np.array([[0.2], [0.7]])
    cov = np.block([[np.eye(1), h.T], [h, np.eye(2) + h @ h.T]])
    with pytest.raises(TimeoutError):
        thin.exact_gauss_thin_pid(
            cov, 1, 1, 1, objective_target=0.5,
            native_stopping=False, timeout=2.0)
    assert received['objective_target'] == 0.5
    assert received['native_stopping'] is False
    assert received['timeout'] == 2.0
