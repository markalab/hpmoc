# hpmoc: HEALPix Multi-Order Coordinate Partial Skymaps

## Installation

`hpmoc` has only a few dependencies, but they are large numerical/scientific
libraries. You should therefore probably create a virtual environment of some
sort before installing. The easiest and best way to do this at the moment is to
use `conda`, which should come with an Anaconda distribution of Python:

```bash
conda create -n hpmoc
conda activate hpmoc
```
note that creating a new environment is optional and hpmoc can now be installed similar to any other python package. 

### With pip

If you just want to use `hpmoc` and don't need to modify the source code, you
can install using pip:

```bash
pip install hpmoc
```

This should install all required dependencies for you.

### Developers

If you want to install from source (to try the latest, unreleased version, or
to make your own modifications, run tests, etc.), first clone the repository:

```bash
git clone git@bitbucket.org:stefancountryman/hpmoc.git
cd hpmoc
```

Make sure the build tool, `flit`, is installed:

```bash
pip install flit
```

Then install an editable version of `hpmoc` with `flit`:

```bash
flit install --symlink
```

As with the `pip` installation method, this should install all requirements for
you. You should now be able to import `hpmoc`. Note that you'll need to quit
your `python` session (or restart the kernel in Jupyter) and reimport `hpmoc`
before your changes to the source code take effect (which is true for any
editable Python installation, FYI).

You can go ahead and run the tests with `pytest` (which should have been
installed automatically by `flit`):

```bash
py.test --doctest-modules --cov=hpmoc
```
