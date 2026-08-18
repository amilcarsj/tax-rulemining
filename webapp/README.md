# Local taxonomy-feature explorer

This separate Flask MVC application visualises the generated taxonomy V2 outputs.
It never reruns or modifies the analysis pipeline.

From the project root:

```bash
.venv/bin/python -m pip install -e .
PYTHONPATH=webapp .venv/bin/python webapp/run.py
```

Open <http://127.0.0.1:5000>.

The application reads:

- `output/<dataset>/taxonomy_v2/step1/`
- `output/<dataset>/taxonomy_v2/step2_top_nodes/`
- `output/<dataset>/taxonomy_v2/step3_meta_patterns/`

The current interface focuses on semantic meta-patterns and feature differences
organized by taxonomy node. It intentionally does not show individual GPS
tracks or trajectory-level map views.
