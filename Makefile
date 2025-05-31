LITSEA_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea") | .version')

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

clean: ## Clean the project
	cargo clean

format: ## Format the project
	cargo fmt

lint: ## Lint the project
	cargo clippy

test: ## Test the project
	cargo test

build: ## Build the project
	cargo build --release

tag: ## Make a new tag for the current version
	git tag v$(LITSEA_VERSION)
	git push origin v$(LITSEA_VERSION)

publish: ## Publish the crate to crates.io
	cargo package && cargo publish
