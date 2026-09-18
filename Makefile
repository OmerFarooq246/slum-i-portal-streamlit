.PHONY: audit check check-text coverage format lint run test typecheck

run:
	python -m streamlit run app.py

format:
	ruff check --fix .
	ruff format .

lint:
	ruff format --check .
	ruff check .

check-text:
	python scripts/check_no_em_dash.py

typecheck:
	mypy

test:
	pytest

coverage:
	pytest --cov --cov-report=term-missing --cov-report=xml

audit:
	python -m pip_audit

check: check-text lint typecheck test
