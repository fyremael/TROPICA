from demos.render_trace_explorer import render_html


def test_trace_explorer_embeds_trace_data():
    traces = [
        {
            "schema_version": 1,
            "scenario": {"provider": "hostile", "suite": "unit"},
            "accepted": True,
            "value": '{"tool":"search","arguments":{}}',
            "parsed": {"tool": "search", "arguments": {}},
            "emitted_token_ids": [1, 2],
            "emitted_text": "{}",
            "steps": 1,
            "events": [
                {
                    "step": 0,
                    "allowed_count": 2,
                    "allowed_token_ids": [1, 2],
                    "selected_token_id": 1,
                    "selected_token_text": "{",
                    "selected_score": 0.5,
                    "selected_was_allowed": True,
                    "top_illegal_token_id": 999,
                    "top_illegal_token_text": None,
                    "top_illegal_score": 1000.0,
                    "accepted": True,
                    "complete_value": '{"tool":"search","arguments":{}}',
                }
            ],
        }
    ]

    html = render_html(traces)

    assert "TROPICA Trace Explorer" in html
    assert "trace-data" in html
    assert "hostile" in html
    assert "top_illegal_score" in html


def test_trace_explorer_handles_unified_support_events():
    traces = [
        {
            "schema_version": 1,
            "trace_type": "unified_support",
            "family": "dyck",
            "scenario": {"provider": "support-contract", "suite": "balanced", "family": "dyck"},
            "accepted": True,
            "value": "( ) <eos>",
            "parsed": None,
            "emitted_token_ids": [],
            "steps": 1,
            "events": [
                {
                    "schema_version": 1,
                    "trace_type": "support_event",
                    "family": "dyck",
                    "scenario": "balanced",
                    "step": 0,
                    "state_summary": {"balance": 0},
                    "planner_support": ["("],
                    "guard_support": ["("],
                    "policy_support": None,
                    "final_support": ["("],
                    "selected": "(",
                    "selected_score": 1.0,
                    "selected_was_allowed": True,
                    "accepting": False,
                    "failure_reason": None,
                    "planner_trace": {},
                }
            ],
        }
    ]

    html = render_html(traces)

    assert "support-contract" in html
    assert "final_support" in html
    assert "selected" in html
