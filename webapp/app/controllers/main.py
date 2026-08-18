"""Controllers for the local taxonomy-feature visual explorer."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for

from app.models.repository import ResultRepository


explorer = Blueprint("explorer", __name__)


def repository() -> ResultRepository:
    if "result_repository" not in current_app.extensions:
        current_app.extensions["result_repository"] = ResultRepository(
            current_app.config["OUTPUT_ROOT"]
        )
    return current_app.extensions["result_repository"]


def common_context(repo: ResultRepository, dataset: str | None = None) -> dict[str, object]:
    datasets = repo.dataset_names()
    selected_dataset = dataset or (datasets[0] if datasets else None)
    return {"datasets": datasets, "selected_dataset": selected_dataset}


@explorer.get("/")
def index():
    repo = repository()
    datasets = repo.dataset_names()
    if not datasets:
        return render_template("empty.html", **common_context(repo))
    selected_dataset = request.args.get("dataset", datasets[0])
    try:
        dashboard = repo.dashboard(selected_dataset)
    except KeyError:
        abort(404)
    return render_template("dashboard.html", dashboard=dashboard, **common_context(repo, selected_dataset))


@explorer.get("/datasets/<dataset>/patterns")
def patterns(dataset: str):
    repo = repository()
    try:
        pattern_rows = repo.patterns(dataset)
    except KeyError:
        abort(404)
    nodes = sorted({node for row in pattern_rows for node in row.get("display_nodes", [])})
    return render_template(
        "patterns.html",
        patterns=pattern_rows,
        nodes=nodes,
        **common_context(repo, dataset),
    )


@explorer.get("/datasets/<dataset>/patterns/<meta_pattern_id>")
def pattern_detail(dataset: str, meta_pattern_id: str):
    repo = repository()
    try:
        detail = repo.pattern_detail(dataset, meta_pattern_id)
    except KeyError:
        abort(404)
    return render_template("pattern_detail.html", detail=detail, **common_context(repo, dataset))


@explorer.get("/datasets/<dataset>/rules")
def rules(dataset: str):
    return redirect(url_for("explorer.patterns", dataset=dataset), code=302)


@explorer.get("/datasets/<dataset>/rules/<rule_id>")
def rule_detail(dataset: str, rule_id: str):
    return redirect(url_for("explorer.patterns", dataset=dataset), code=302)


@explorer.get("/api/datasets/<dataset>/summary")
def api_summary(dataset: str):
    try:
        return jsonify(repository().dashboard(dataset))
    except KeyError:
        abort(404)


@explorer.get("/api/datasets/<dataset>/patterns")
def api_patterns(dataset: str):
    try:
        return jsonify(repository().patterns(dataset))
    except KeyError:
        abort(404)


@explorer.get("/api/datasets/<dataset>/patterns/<meta_pattern_id>")
def api_pattern_detail(dataset: str, meta_pattern_id: str):
    try:
        return jsonify(repository().pattern_detail(dataset, meta_pattern_id))
    except KeyError:
        abort(404)
