"""Run one motor-policy tick, from numeric weights or a small Python subset."""
from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable


FEATURES = (
    "1", "q0", "q1", "dq0", "dq1", "x", "y", "vx", "vy",
    "goal_x-x", "goal_y-y", "prev_u0", "prev_u1", "prev2_u0", "prev2_u1",
)
OBS_FIELDS = ("q", "dq", "tip", "tip_v", "prev_u", "prev2_u")
MAX_TORQUE = 0.8
MAX_HIDDEN = 16
MAX_SOURCE = 8000
MAX_NODES = 768


class PolicyError(ValueError):
    """The submitted module is malformed or cannot evaluate on this observation."""


def _number(x, label="value"):
    if type(x) not in (int, float):
        raise PolicyError(f"{label} must be a number")
    try:
        y = float(x)
    except OverflowError as exc:
        raise PolicyError(f"{label} is too large") from exc
    if not math.isfinite(y):
        raise PolicyError(f"{label} must be finite")
    return y


def _vector(x, size, label):
    if not isinstance(x, (list, tuple)) or len(x) != size:
        raise PolicyError(f"{label} must contain {size} numbers")
    return [_number(i, label) for i in x]


def _inputs(obs, goal):
    if not isinstance(obs, dict) or any(key not in obs for key in OBS_FIELDS):
        raise PolicyError("observation is missing a policy input")
    return {key: _vector(obs[key], 2, key) for key in OBS_FIELDS}, _vector(goal, 2, "goal")


def _output(value):
    return [min(MAX_TORQUE, max(-MAX_TORQUE, u)) for u in _vector(value, 2, "motor output")]


def _compile_weights(spec):
    if not isinstance(spec, dict) or set(spec) != {"layers"}:
        raise PolicyError("weights must be an object with only a 'layers' field")
    layers = spec["layers"]
    if not isinstance(layers, list) or len(layers) not in (1, 2):
        raise PolicyError("supply one output layer, or one hidden layer and one output layer")

    compiled = []
    n_in = len(FEATURES)
    for i, layer in enumerate(layers):
        final = i == len(layers) - 1
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise PolicyError(f"layer {i} needs 'weight' and 'bias'")
        weights = layer["weight"]
        if not isinstance(weights, list):
            raise PolicyError(f"layer {i} weight must be a matrix of rows")
        n_out = len(weights)
        if final and n_out != 2:
            raise PolicyError("the output layer must have exactly 2 rows")
        if not final and not (1 <= n_out <= MAX_HIDDEN):
            raise PolicyError(f"the hidden layer must have 1–{MAX_HIDDEN} rows")
        compiled.append((
            [_vector(row, n_in, f"layer {i} weight row") for row in weights],
            _vector(layer["bias"], n_out, f"layer {i} bias"),
        ))
        n_in = n_out

    def run(obs, goal):
        obs, goal = _inputs(obs, goal)
        x, y = obs["tip"]
        z = [
            1.0, *obs["q"], *obs["dq"], x, y, *obs["tip_v"],
            goal[0] - x, goal[1] - y, *obs["prev_u"], *obs["prev2_u"],
        ]
        try:
            for i, (weights, bias) in enumerate(compiled):
                z = [
                    math.fsum(w * a for w, a in zip(row, z)) + b
                    for row, b in zip(weights, bias)
                ]
                z = _vector(z, len(z), f"layer {i} output")
                if i < len(compiled) - 1:
                    z = [math.tanh(v) for v in z]
        except (OverflowError, ValueError) as exc:
            raise PolicyError(f"weights could not be evaluated: {exc}") from exc
        return _output(z)

    return run


_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: math.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_MATH = {
    "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "atan2": math.atan2, "hypot": math.hypot,
    "exp": math.exp, "log": math.log, "tanh": math.tanh,
}
_CALLS = {*_MATH, "min", "max"}
_SYNTAX = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AugAssign, ast.If, ast.Expr, ast.Pass, ast.Name,
    ast.Constant, ast.List, ast.Tuple, ast.Subscript, ast.Load, ast.Store,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.And, ast.Or, ast.Not, *_BINARY, *_UNARY, *_COMPARE,
}


def _error(node, message):
    line = getattr(node, "lineno", None)
    return PolicyError(f"line {line}: {message}" if line is not None else message)


def _parse(source):
    if not isinstance(source, str) or not source.strip():
        raise PolicyError("code must be a Python source string defining policy(obs, goal)")
    if len(source) > MAX_SOURCE:
        raise PolicyError(f"policy source must be at most {MAX_SOURCE} characters")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise PolicyError(f"invalid policy syntax: {exc}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise PolicyError(f"policy source exceeds {MAX_NODES} syntax nodes")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise PolicyError("the only top-level statement must be def policy(obs, goal):")
    fn = tree.body[0]
    args = fn.args
    if (fn.name != "policy" or [x.arg for x in args.args] != ["obs", "goal"]
            or args.posonlyargs or args.kwonlyargs or args.defaults or args.vararg
            or args.kwarg or fn.decorator_list or fn.returns
            or getattr(fn, "type_params", ()) or any(x.annotation for x in args.args)):
        raise _error(fn, "use exactly def policy(obs, goal): without annotations or decorators")
    for node in nodes:
        if type(node) not in _SYNTAX or (isinstance(node, ast.FunctionDef) and node is not fn):
            raise _error(node, f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                          or node.func.id not in _CALLS or node.keywords):
            raise _error(node, f"only these direct calls are allowed: {', '.join(sorted(_CALLS))}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise _error(node, "names starting with '_' are not allowed")
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                _number(node.value, "literal")
            elif type(node.value) not in (str, bool):
                raise _error(node, "only numeric, string and boolean literals are allowed")
        if isinstance(node, ast.Return) and node.value is None:
            raise _error(node, "return a two-element motor vector")
        if isinstance(node, ast.Assign) and any(
                not isinstance(target, (ast.Name, ast.Tuple, ast.List)) for target in node.targets):
            raise _error(node, "assign to local variables, not into a vector or observation")
        if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
            raise _error(node, "augmented assignment needs a local variable")
    return fn.body


def _operation(op, x, y):
    return _number(_BINARY[type(op)](_number(x), _number(y)), "arithmetic result")


def _expr(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id not in env:
            raise _error(node, f"unknown variable '{node.id}'")
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_expr(x, env) for x in node.elts]
    if isinstance(node, ast.Subscript):
        value, key = _expr(node.value, env), _expr(node.slice, env)
        if isinstance(value, dict) and type(key) is str:
            return value[key]
        if isinstance(value, list) and type(key) in (int, float) and key == int(key):
            return value[int(key)]
        raise _error(node, "use an observation field name or an integer vector index")
    if isinstance(node, ast.BinOp):
        return _operation(node.op, _expr(node.left, env), _expr(node.right, env))
    if isinstance(node, ast.UnaryOp):
        value = _expr(node.operand, env)
        return not value if isinstance(node.op, ast.Not) else _UNARY[type(node.op)](_number(value))
    if isinstance(node, ast.IfExp):
        return _expr(node.body if _expr(node.test, env) else node.orelse, env)
    if isinstance(node, ast.BoolOp):
        result = _expr(node.values[0], env)
        for expr in node.values[1:]:
            if bool(result) == isinstance(node.op, ast.Or):
                break
            result = _expr(expr, env)
        return result
    if isinstance(node, ast.Compare):
        left = _expr(node.left, env)
        for op, expr in zip(node.ops, node.comparators):
            right = _expr(expr, env)
            if type(left) not in (int, float, bool, str) or type(right) not in (int, float, bool, str):
                raise _error(node, "compare scalar values, not entire vectors or observations")
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        values = [_expr(x, env) for x in node.args]
        name = node.func.id
        if name in {"min", "max"}:
            if len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            return (min if name == "min" else max)(_number(x) for x in values)
        return _number(_MATH[name](*[_number(x) for x in values]), "function result")
    raise _error(node, f"{type(node).__name__} is not an expression")


def _assign(target, value, env):
    if isinstance(target, ast.Name):
        if target.id in {"obs", "goal", "pi", *_CALLS}:
            raise _error(target, f"'{target.id}' is reserved")
        env[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        if not isinstance(value, list) or len(value) != len(target.elts):
            raise _error(target, "unpacking requires a vector with matching length")
        for part, element in zip(target.elts, value):
            _assign(part, element, env)
    else:
        raise _error(target, "assign to local variables only")


def _statements(body, env):
    for node in body:
        if isinstance(node, ast.Return):
            return True, _expr(node.value, env)
        if isinstance(node, ast.Assign):
            value = _expr(node.value, env)
            for target in node.targets:
                _assign(target, value, env)
        elif isinstance(node, ast.AugAssign):
            result = _operation(node.op, _expr(node.target, env), _expr(node.value, env))
            _assign(node.target, result, env)
        elif isinstance(node, ast.If):
            returned, value = _statements(node.body if _expr(node.test, env) else node.orelse, env)
            if returned:
                return True, value
        elif isinstance(node, ast.Expr):
            _expr(node.value, env)
    return False, None


def _compile_code(source):
    body = _parse(source)

    def run(obs, goal):
        obs, goal = _inputs(obs, goal)
        try:
            returned, value = _statements(body, {"obs": obs, "goal": goal})
        except PolicyError:
            raise
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise PolicyError(f"policy evaluation failed: {exc}") from exc
        if not returned:
            raise PolicyError("the executed policy branch did not return")
        return _output(value)

    return run


def compile_policy(arm: str, spec: object) -> Callable[[dict, list[float]], list[float]]:
    """Validate an artifact and return a stateless motor-command function."""
    if arm == "weights":
        return _compile_weights(spec)
    if arm == "code":
        return _compile_code(spec)
    raise PolicyError("arm must be 'weights' or 'code'")
