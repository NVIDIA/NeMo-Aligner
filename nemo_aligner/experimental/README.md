# Experimental Package

The `experimental` sub-package contains projects that are under active development and may not be fully stable.

## Experimental Project Directory Structure:

```
NeMo-Aligner/
├── docs/
│   ├── user-guide/
│   │   └── ppo.html
│   └── user-guide-experimental/    <----- experimental docs
│       └── new-thing.html
├── nemo_aligner/
│   ├── algorithms/
│   ├── data/
│   │   ├── datasets.py
│   │   └── tests/
│   │       └── datasets_test.py
│   └── experimental/               <----- experimental sub-package
│       ├── <proj-name>/
│           ├── dataset.py          <----- experimental dataset
│           ├── new_algo.py         <----- experimental algo
│           ├── model.py            <----- experimental model
│           └── tests/
│               └── model_test.py   <----- experimental model test
└── tests/
    └── functional/
        └── dpo.sh
        └── test_cases/
            └── dpo-llama3
    └── functional_experimental/    <----- experimental functional tests (mirrors functional/ structure)
        ├── new_algo.sh
        └── test_cases/
            └── new_algo-llama3
```

The directories below exist to organize experimental projects (source code), tests, and documentation.

- [nemo_aligner/experimental/](../../nemo_aligner/experimental/): Main experimental sub-package containing projects under development
- [tests/functional_experimental/](../../tests/functional_experimental/): Functional tests for experimental projects
- [docs/user-guide-experimental/](../../docs/user-guide-experimental/): Documentation directory for experimental features and algorithms

The `experimental` sub-package follows a modular structure where each project has its own directory (sub-package) containing implementation and tests.

## Guidelines for "experimental/" Projects

- **Scope**: Projects can include new model definitions, training loops, utilities, or unit tests.
- **Independence**: Projects should ideally be independent. Dependence on other projects signals it might benefit from being added to core with tests (and documentation if applicable).
- **Testing**: Must include at least one functional test [example](../../tests/functional/test_cases/dpo-llama3).
