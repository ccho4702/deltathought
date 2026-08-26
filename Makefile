.PHONY: setup lint test sanity verify provenance data-audit annotation-audit backbone-smoke language-smoke report verify-all

setup:
	uv sync --group dev

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

sanity:
	uv run deltaomni-sanity --config configs/sanity.yaml

verify:
	uv run deltaomni-verify --config configs/sanity.yaml

provenance:
	uv run deltaomni-provenance --config configs/provenance.yaml

data-audit:
	uv run deltaomni-data-audit --allow-not-ready

annotation-audit:
	uv run deltaomni-annotation-audit

backbone-smoke:
	uv run deltaomni-backbone-smoke

language-smoke:
	uv run deltaomni-language-smoke

report:
	uv run deltaomni-report

verify-all: lint test provenance data-audit annotation-audit verify backbone-smoke language-smoke report
