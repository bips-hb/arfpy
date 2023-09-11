# Contributing to `arfpy`

We welcome contributions from the community! We greatly appreciate any effort that helps making `arfpy` a transparent and valuable open source project. This guide is based on this [template](https://github.com/NLeSC/python-template/blob/main/CONTRIBUTING.md) and explains how you can contribute to `arfpy`.


A contribution can be one of the following cases:

1. You have a question;
1. You think you may have found a bug (including unexpected behavior);
1. You want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation).

The sections below outline the steps in each case.


## You have a question

1. Browse through the reported issues [here](https://github.com/bips-hb/arfpy/issues) to see if someone already filed the same issue;
1. If your issue search did not yield any relevant results, please open a new issue;
1. Add a relevant label to your issue, e.g. "Question"

## You think you may have found a bug

1. Browse through the reported issues [here](https://github.com/bips-hb/arfpy/issues) to see if someone already filed the same issue;
1. If your issue search did not yield any relevant results, please open a new issue. Make sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you should include:
    - a reproducible example
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
1. Add relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. It may be a good idea to announce your plan to the rest of the community _before you start working_ by opening a new issue
1. If needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
1. Make sure that all relevant dependencies specified in the [requirements.txt](https://github.com/bips-hb/arfpy/blob/master/requirements.txt) are installed; it might be a good idea to use a [conda environment](https://conda.io/activation) at this stage;
1. Make sure the existing automated tests run. To do so, navigate to the [tests](https://github.com/bips-hb/arfpy/tree/master/tests) folder and run `` python3 test.py`` in your command line;
1. Add your own tests (if necessary) to the [tests](https://github.com/bips-hb/arfpy/tree/master/tests) folder;
1. Update or expand the documentation in the [docs](https://github.com/bips-hb/arfpy/tree/master/docs) folder;
1. Push your feature branch to (your fork of) the repository on GitHub;
1. Create a [pull request](https://help.github.com/articles/creating-a-pull-request/)

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request! Please try to tick off as many points as you can (this helps making the process faster, because the `arfpy` package maintainers may be quite busy) and submit the pull request. We'll do our best to help, but keep in mind that you might be asked to append additional commits to your pull request.
