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
