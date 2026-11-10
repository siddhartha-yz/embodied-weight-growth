"""Replay one original absolute-weight proposal without the editor or SDK."""

from weight_policy import HIDDEN_NAMES, normalize_checkpoint, numeric


def apply_proposal(current, proposal, feature_names, action_names):
    if not isinstance(proposal, dict) or set(proposal) != {"rows", "diagnosis"}:
        raise ValueError("Proposal must contain rows and diagnosis")
    if not isinstance(proposal["rows"], list) or not isinstance(proposal["diagnosis"], str):
        raise ValueError("rows must be an array and diagnosis must be text")
    before = normalize_checkpoint(current, feature_names, action_names)
    updated = normalize_checkpoint(current, feature_names, action_names)
    seen = set()
    changed = 0
    for edit in proposal["rows"]:
        if not isinstance(edit, dict) or set(edit) != {"to", "sources", "values"}:
            raise ValueError("A row must contain to, sources, and values")
        target = edit["to"]
        if not isinstance(target, str) or target in seen:
            raise ValueError(f"Invalid or duplicate target {target!r}")
        seen.add(target)
        if target in HIDDEN_NAMES:
            layer, allowed = "hidden", set(feature_names)
        elif target in action_names:
            layer, allowed = "outputs", {*feature_names, *HIDDEN_NAMES}
        else:
            raise ValueError(f"Unknown row {target!r}")
        sources, values = edit["sources"], edit["values"]
        if (
            not isinstance(sources, list)
            or not isinstance(values, list)
            or not sources
            or len(sources) != len(values)
            or not all(isinstance(k, str) for k in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError(f"{target}: expected equally sized sources/values arrays and no repeated source")
        row = updated[layer].setdefault(target, {})
        for source, value in zip(sources, values, strict=True):
            value = numeric(value, f"{target}.{source}")
            if source == "bias":
                changed += value != row.get("bias", 0)
                row["bias"] = value
            else:
                if source not in allowed:
                    raise ValueError(f"{target}: unknown source {source!r}")
                weights = row.setdefault("weights", {})
                changed += value != weights.get(source, 0)
                weights[source] = value
    result = normalize_checkpoint(updated, feature_names, action_names)
    if result == before:
        raise ValueError("Proposal does not change any numeric coefficient")
    return result, changed
