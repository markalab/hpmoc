# hpmoc: HEALPix Multi-Order Coordinate Partial Skymaps

HPMOC is an ultra high-performance, cross-platform toolset for working with
multi-order coordinate (MOC) HEALPix_ images (i.e. images with multiple pixel
resolutions). MOC images are used by the [LIGO-Virgo-KAGRA collaboration](https://www.ligo.org/),
the [Interplanerary Network](https://ipn3.ssl.berkeley.edu/) and others to represent portions of
the sky with variable resolution. By only including pixels in regions of
interest, and only then at a resolution appropriate to how they were
observed/calculated, it is possible to reduce storage and computation costs by
several orders of magnitude.

HPMOC is the *only* library providing tools for loading partial/whole MOC
skymaps (as well as standard HEALPix skymaps), taking spatial intersections,
modifying resolution, plotting the skymaps, converting them to and from
[Astropy WCS](https://docs.astropy.org/en/stable/wcs/index.html) projections,
performing pointwise math, and generating PSF skymaps from point sources,
all using algorithms that minimize memory, computation, and storage costs.
It is based off of work on [LLAMA](https://multimessenger.science), the world's first Gravitational
Wave/High-Energy Neutrino low-latency search pipeline, which has been improved
and refactored into this separate module.

If you use `hpmoc` in published research, we ask that you cite [Stefan Countryman's thesis](https://academiccommons.columbia.edu/doi/10.7916/c8n9-p112).
`hpmoc` is introduced in section 4.5.13.

`hpmoc` is licensed under the terms of the [GNU General Public License, version 2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## Installation

`hpmoc` has only a few dependencies, but they are large numerical/scientific
libraries. You should therefore probably create a virtual environment of some
sort before installing. The easiest and best way to do this at the moment is to
use `conda`, which should come with an Anaconda distribution of Python:

```bash
conda create -n hpmoc
conda activate hpmoc
```

### With pip

If you just want to use `hpmoc` and don't need to modify the source code, you
can install [the last released version](https://github.com/markalab/hpmoc/releases/latest)
using pip:

```bash
pip install hpmoc-<REPLACE_WITH_VERSION>-py3-none-any.whl
```

This should install all required dependencies for you.

### Developers

If you want to install from source (to try the latest, unreleased version, or
to make your own modifications, run tests, etc.), first clone the repository:

```bash
git clone https://github.com/markalab/hpmoc.git
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
