"""Run pytest-style test modules in an env without pytest (the sudoku / sudoku-fine envs).

    python tools/pytest_shim.py test_casper_sudoku_layer.py [more test files ...]

A minimal stand-in `pytest` module is installed in sys.modules for this
process only (a real pytest, if any, is not touched). It covers what the
Sudoku tests use: raises(match=), mark.parametrize, mark.skipif, param(marks=),
skip, fixture (module scope), and the tmp_path fixture. Prints one line per
test and a pytest-like summary; the exit code is 1 if anything failed.
"""
import importlib.util
import inspect
import itertools
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path


class _Skipped(Exception):
    pass


class _Param:
    def __init__(self, *values, marks=(), id=None):
        self.values = values
        self.marks = marks if isinstance(marks, (list, tuple)) else (marks,)
        self.id = id


class _SkipIf:
    def __init__(self, condition, reason=''):
        self.condition = bool(condition)
        self.reason = reason

    def __call__(self, fn):
        fn.__dict__.setdefault('_shim_skips', []).append(self)
        return fn


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(',')] if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            fn.__dict__.setdefault('_shim_params', []).insert(0, (names, list(argvalues)))
            return fn
        return deco

    @staticmethod
    def skipif(condition, reason=''):
        return _SkipIf(condition, reason)


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, etype, value, tb):
        if etype is None:
            raise AssertionError(f'DID NOT RAISE {self.exc}')
        if not issubclass(etype, self.exc):
            return False
        if self.match is not None and not re.search(self.match, str(value)):
            raise AssertionError(f'{self.match!r} does not match {str(value)!r}')
        return True


def _fixture(fn=None, **_kwargs):
    def deco(f):
        f._shim_fixture = True
        return f
    return deco(fn) if fn is not None else deco


def _skip(reason=''):
    raise _Skipped(reason)


def _install_stub():
    stub = types.ModuleType('pytest')
    stub.raises = _Raises
    stub.mark = _Mark()
    stub.param = _Param
    stub.fixture = _fixture
    stub.skip = _skip
    stub.__shim__ = True
    sys.modules['pytest'] = stub


def _cases(fn):
    """Expand stacked parametrize marks into (id, kwargs, skip-marks) tuples."""
    layers = fn.__dict__.get('_shim_params', [])
    if not layers:
        yield '', {}, []
        return
    expanded = []
    for names, values in layers:
        rows = []
        for v in values:
            p = v if isinstance(v, _Param) else _Param(*(v if len(names) > 1 else (v,)))
            rows.append((p.id or '-'.join(str(x) for x in p.values),
                         dict(zip(names, p.values)), list(p.marks)))
        expanded.append(rows)
    for combo in itertools.product(*expanded):
        yield ('-'.join(c[0] for c in combo),
               {k: v for c in combo for k, v in c[1].items()},
               [m for c in combo for m in c[2]])


def run_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)

    fixtures = {n: f for n, f in vars(mod).items() if getattr(f, '_shim_fixture', False)}
    fixture_values = {}
    counts = {'passed': 0, 'failed': 0, 'skipped': 0}
    for name, fn in list(vars(mod).items()):
        if not (name.startswith('test_') and inspect.isfunction(fn)):
            continue
        for case_id, kwargs, marks in _cases(fn):
            label = f'{path.name}::{name}' + (f'[{case_id}]' if case_id else '')
            skips = [m for m in fn.__dict__.get('_shim_skips', []) + marks
                     if isinstance(m, _SkipIf) and m.condition]
            if skips:
                counts['skipped'] += 1
                print(f'SKIPPED {label} ({skips[0].reason})')
                continue
            try:
                call = dict(kwargs)
                for arg in inspect.signature(fn).parameters:
                    if arg in call:
                        continue
                    if arg == 'tmp_path':
                        call[arg] = Path(tempfile.mkdtemp(prefix='shim_'))
                    elif arg in fixtures:
                        if arg not in fixture_values:
                            fixture_values[arg] = fixtures[arg]()
                        call[arg] = fixture_values[arg]
                    else:
                        raise TypeError(f'unknown fixture {arg!r}')
                fn(**call)
            except _Skipped as exc:
                counts['skipped'] += 1
                print(f'SKIPPED {label} ({exc})')
            except Exception:
                counts['failed'] += 1
                print(f'FAILED  {label}')
                traceback.print_exc()
            else:
                counts['passed'] += 1
                print(f'PASSED  {label}')
    return counts


def main(argv):
    _install_stub()
    print('[shim] using the minimal pytest stand-in')
    total = {'passed': 0, 'failed': 0, 'skipped': 0}
    for test_file in argv:
        for k, v in run_module(test_file).items():
            total[k] += v
    n = sum(total.values())
    print(f"== {total['passed']} passed, {total['failed']} failed, "
          f"{total['skipped']} skipped ({n} collected) ==")
    return 1 if total['failed'] else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
