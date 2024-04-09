# Contributions are welcome!

Thanks for the interest in contributing to NeMo-Aligner. We do all of NeMo-Aligner's development in the open. Contributions from the community are welcome.

# Pull Requests (PR) Guidelines

**Send your PRs to the `main` or `dev` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
2) Read General Principles and style guide below
3) Make sure you sign your commits. E.g. use ``git commit -sS`` when committing.
4) Make sure all unittests finish successfully before sending PR ``pytest`` or (if your dev box does not have GPU) ``pytest --cpu`` from the root folder
5) Send your PR and request a review

**NOTE**: The `main` branch uses a fixed NeMo version which we will update on every release. The `dev` branch is the branch that has all commits from `main` but uses NeMo's main branch: this branch is less stable but we run nightly tests on it to make sure everything works. We only provide the dockerfile that works with `main`, which is the branch most PRs should target unless they require the latest NeMo main (in which case they should target `dev`).

Every release `dev` and `main` will sync to be the same.

## Unit tests
Quick unit tests (locally, while developing)
```
pytest
# If you don't have NVIDIA GPU do:
# pytest --cpu
```

## Whom should you ask for review:

@gshennvm or @odelalleau

Your pull requests must pass all checks and peer-review before they can be merged. For certain changes, we will manually trigger convergence tests.

# General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Well-tested**: please add simple, fast unittests. Consider adding CI tests for end-to-end functionality.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo-Aligner supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Python style
We use ``black`` as our style guide. To fix your format run `pip install pre-commit && pre-commit install && pre-commit run --all`.

1. Include docstrings for every class and method exposed to the user.
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Add ``__init__.py`` for every folder.
1. F-strings are prefered to formatted strings.
1. Loggers are preferred to print. Use the logger from NeMo via ``from nemo.utils import logging``
1. Private functions (functions start with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.

# Algorithms
Algorithms is a grouping of trainers that coordinate training. Before contributing see if you can use an existing algorithm implementation for your use case.

Thank you for contributing to NeMo-Aligner!
