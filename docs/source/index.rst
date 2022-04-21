.. HEALPix Multi-Order Coordinate Library (HPMOC) documentation master file, created by
   sphinx-quickstart on Fri Dec 31 14:28:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to HEALPix Multi-Order Coordinate Library (HPMOC)'s documentation!
==========================================================================

HPMOC is an ultra high-performance, cross-platform toolset for working with
multi-order coordinate (MOC) HEALPix_ images (i.e. images with multiple pixel
resolutions). MOC images are used by LIGO_ and others to represent portions of
the sky with variable resolution. By only including pixels in regions of
interest, and only then at a resolution appropriate to how they were
observed/calculated, it is possible to reduce storage and computation costs by
several orders of magnitude.

HPMOC is the *only* library providing tools for loading partial/whole MOC
skymaps (as well as standard HEALPix skymaps), taking spatial intersections,
modifying resolution, plotting the skymaps, converting them to and from
`Astropy WCS`_ projections, performing pointwise math, and generating PSF
skymaps from point sources, all using algorithms that minimize memory,
computation, and storage costs. It is based off of work on LLAMA_, the world's
first Gravitational Wave/High-Energy Neutrino low-latency search pipeline,
which has been improved and refactored into this separate module.

If you are interested in using HPMOC for your research, or if you have
questions about it, reach out to `the author`_.

.. _HEALPix: https://healpix.sourceforge.io/html/intro.htm
.. _LIGO: https://www.ligo.org/
.. _`Astropy WCS`: https://docs.astropy.org/en/stable/wcs/index.html
.. _LLAMA: https://multimessenger.science
.. _`the author`: https://stc.sh

Installation
------------

``hpmoc`` has only a few dependencies, but they are large numerical/scientific
libraries. You should therefore probably create a virtual environment of some
sort before installing. The easiest and best way to do this at the moment is to
use ``conda``, which should come with an Anaconda distribution of Python:

.. code:: bash

    conda create -n hpmoc
    conda activate hpmoc

With pip
~~~~~~~~

If you just want to use ``hpmoc`` and don't need to modify the source code, you
can install the last released version (link in source repo README) using pip:

.. code:: bash

   pip install hpmoc-latest-py3-none-any.whl

This should install all required dependencies for you.

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

If you want to install from source (to try the latest, unreleased version, or
to make your own modifications, run tests, etc.), first clone the repository:

.. code:: bash

    git clone git@bitbucket.org:stefancountryman/hpmoc.git
    cd hpmoc

Make sure the build tool, ``flit``, is installed:

.. code:: bash

    pip install flit

Then install an editable version of ``hpmoc`` with ``flit``:

.. code:: bash

    flit install --symlink

As with the ``pip`` installation method, this should install all requirements for
you. You should now be able to import ``hpmoc``. Note that you'll need to quit
your ``python`` session (or restart the kernel in Jupyter) and reimport ``hpmoc``
before your changes to the source code take effect (which is true for any
editable Python installation, FYI).

You can go ahead and run the tests with ``pytest`` (which should have been
installed automatically by ``flit``):

.. code:: bash

    py.test --doctest-modules --cov=hpmoc

.. toctree::
   :maxdepth: 6
   :caption: Examples

   jup/plotting-examples.ipynb

.. toctree::
   :maxdepth: 6
   :caption: API Documentation

   hpmoc

.. include:: hpmoc-subcomponents.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
