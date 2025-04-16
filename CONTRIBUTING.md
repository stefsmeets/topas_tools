# Contributing guidelines

Welcome! *topas_tools* is an open-source project for analysis of molecular dynamics simulations. If you're trying *topas_tools* with your data, your experience, questions, bugs you encountered, and suggestions for improvement are important to the success of the project.

We have a [Code of Conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Questions, feedback, bugs

Use the search function to see if someone else already ran accross the same issue. Feel free to open a new [issue here](https://github.com/stefsmeets/topas_tools/issues) to ask a question, suggest improvements/new features, or report any bugs that you ran into.

## Submitting changes

Even better than a good bug report is a fix for the bug or the implementation of a new feature. We welcome any contributions that help improve the code.

When contributing to this repository, please first discuss the change you wish to make via an [issue](https://github.com/stefsmeets/topas_tools/issues) with the owners of this repository before making a change.

Contributions can come in the form of:

- Bug fixes
- New features
- Improvement of existing code
- Updates to the documentation
- ... ?

We use the usual GitHub pull-request flow. For more info see [GitHub's own documentation](https://help.github.com/articles/using-pull-requests/).

Typically this means:

1. [Forking](https://docs.github.com/articles/about-forks) the repository and/or make a [new branch](https://docs.github.com/articles/about-branches)
2. Making your changes
3. Make sure that the tests pass and add your own
4. Update the documentation is updated for new features
5. Pushing the code back to Github
6. [Create a new Pull Request](https://help.github.com/articles/creating-a-pull-request/)

One of the code owners will review your code and request changes if needed. Once your changes have been approved, your contributions will become part of *topas_tools*. 🎉

## Getting started with development

### Setup

*topas_tools* targets Python 3.9 or newer.

Clone the repository into the `topas_tools` directory:

```console
git clone https://github.com/stefsmeets/topas_tools
```

Install using `conda`:

```console
conda create -n topas_tools -c conda-forge cctbx-base
conda activate topas_tools
pip install -e .[develop]
```

### Making a release

The versioning scheme we use is [SemVer](http://semver.org/).

1. Bump and commit the version (`major`/`minor`/`patch` as needed)

```console
bumpversion minor
```

2. Build and upload to pypi

```console
python -m build
twine upload dist/*
```

3. Make a release on Github
