format:
	black .
	isort .

test-format:
	black . --check
	isort . --check-only
	env PYTHONPATH=. pytest --pylint --flake8 -W ignore::DeprecationWarning
