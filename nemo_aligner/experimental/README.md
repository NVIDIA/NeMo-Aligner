# Experimental Package

The `experimental` sub-package contains projects that are under active development and may not be fully stable.

## Experimental Project Directory Structure:

The directory structure above shows the key experimental components:

- [docs/user-guide-experimental/](../../docs/user-guide-experimental/): Documentation directory for experimental features and algorithms
- [nemo_aligner/experimental/](../../nemo_aligner/experimental/): Main experimental sub-package containing projects under development

The `experimental` sub-package follows a modular structure where each project has its own directory (sub-package) containing implementation and tests.

## Guidelines for "experimental/" Projects

- **Scope**: Projects can include new model definitions, training loops, utilities, or unit tests.
- **Independence**: Projects should ideally be independent. Dependence on other projects signals it might benefit from being added to core with tests (and documentation if applicable).
- **Testing**: Must include at least one functional test [example](../../tests/functional/test_cases/dpo-llama3).
