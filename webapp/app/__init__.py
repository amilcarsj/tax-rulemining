"""Flask application factory for the local visual explorer."""

from __future__ import annotations

from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        PROJECT_ROOT=Path(__file__).resolve().parents[2],
        OUTPUT_ROOT=Path(__file__).resolve().parents[2] / "output",
    )

    from app.controllers.main import explorer

    app.register_blueprint(explorer)
    return app
