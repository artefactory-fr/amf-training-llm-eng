.DEFAULT_GOAL = help

# help: help                                   - Display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: install                                - Create a virtual environment and install dependencies
.PHONY: install
install:
	@uv sync
	@make install_precommit

# help: install_precommit                      - Install pre-commit hooks
.PHONY: install_precommit
install_precommit:
	@uv run pre-commit install -t pre-commit
	@uv run pre-commit install -t pre-push

# help: run_app                        - Run streamlit app
.PHONY: run_app
run_app:
	@uv run streamlit run app/main.py --server.runOnSave=true
