from cdsd.planners.json_schema import JSONSchemaGuard, JSONSchemaPlanner, JSONSchemaSpec, JSONSchemaState, render_json_tokens


def test_json_schema_subset_exact_enum_object():
    spec = JSONSchemaSpec.enum_object({"color": ["red", "blue"], "size": ["S", "M"]})
    planner = JSONSchemaPlanner(spec)
    guard = JSONSchemaGuard(spec)
    state = JSONSchemaState()
    tokens = []
    for expected in ["{", "color", ":", "red", ",", "size", ":", "M", "}", "<eos>"]:
        pout = planner.step(state)
        gmask = guard.mask(tokens, state)
        assert expected in pout.plan_mask.allowed
        assert expected in gmask.allowed
        tokens.append(expected)
        state = guard.update(state, expected)
    assert render_json_tokens(tokens) == '{"color":"red","size":"M"}'
