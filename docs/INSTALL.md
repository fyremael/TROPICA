# Install

## Editable install

```bash
python -m pip install -e ".[real-tokenizers,dev]"
cdsd-report --with-pytest --artifacts artifacts --jobs 4
```

The report command writes CSV summaries, SVG dashboards, `report_index.md`, and
`report_manifest.json` to the artifact directory.

## Build a wheel

```bash
python -m build
```

## Smoke-test a wheel

```bash
python -m pip install --force-reinstall "dist/control_delta_support_decoding-0.1.0-py3-none-any.whl[real-tokenizers]"
python -c "import cdsd; print(cdsd.__all__)"
mkdir ../cdsd-wheel-smoke
cd ../cdsd-wheel-smoke
cdsd-report --artifacts smoke_artifacts --jobs 4
```

Run the smoke command from outside the source checkout when you want to prove
the installed wheel carries the report modules on its own. Use `--with-pytest`
only from a source checkout that contains `tests/`.

Use `--jobs 1` for serial logs, or increase `--jobs` to run independent report
tracks in parallel.
