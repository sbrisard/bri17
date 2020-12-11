************
Installation
************


Installing the C++ library
==========================

bri17 has the following dependencies, required for the tests only

- `Eigen <http://eigen.tuxfamily.org/>`_
- `FFTW <http://www.fftw.org/>`_

This is a CMake_ based project. The installation procedure is standard. First,
clone the repository. Then, ``cd`` into the root directory of the bri17
project. Let ``bri17_INSTALL_PREFIX`` be the path to the directory where bri17
should be installed::

  $ git clone https://github.com/sbrisard/bri17
  $ cd bri17
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=bri17_INSTALL_PREFIX ..
  $ cmake --build . --config Release
  $ cmake --install . --config Release

.. note:: The ``--config`` option might not be available, depending on the
   selected generator.

At this point, bri17 should be installed. You can now run the tests::

  $ ctest . -C Release

.. note:: Depending on the system, you might need to add
   ``bri17_INSTALL_PREFIX`` to your ``PATH`` environment variable.


Compiling your first bri17 program
==================================

In this section, we create a simple example that instantiates a
:cpp:class:`bri17::CartesianGrid` and prints it to ``stdout``. The layout of
this project is::

  example/
    ├─ src/
    │   └─ bri17_example.cc
    └─ CMakeLists.txt

And the files are listed below.

:download:`bri17_example.cc <../example/src/bri17_example.cc>`

.. literalinclude:: ../example/src/bri17_example.cc

:download:`CMakeLists.txt <../example/CMakeLists.txt>`

.. literalinclude:: ../example/CMakeLists.txt
   :language: cmake

``cd`` into the ``example`` subdirectory. The provided example program should be
compiled and linked against bri17::

  $ mkdir build
  $ cd build
  $ cmake -Dbri17_DIR=bri17_INSTALL_PREFIX/lib/cmake/bri17 ..
  $ cmake --build . --config Release

An executable called ``example_bri17`` should be present in the
``build/Release`` subdirectory. Running the generated program produces the
following output::

  $ ./bri17_example.exe
  CartesianGrid<3>={L=[1,2,3,],N=[3,4,5,]}


Building the documentation
==========================

The documentation of bri17 requires Sphinx_. The C++ API docs are built with
Doxygen_ and the Breathe_ extension to Sphinx_.

To build the HTML version of the docs in the ``public`` subdirectory::

  $ cd docs
  $ sphinx-build -b html . ../public

To build the LaTeX version of the docs::

  $ cd docs
  $ make latex


Installing the Python bindings
==============================

To install the bri17 module, ``cd`` into the ``python`` subdirectory and edit
the ``setup.cfg`` file. Set the ``include_dir`` and ``library_dir`` to the
appropriate paths. These should be::

  [bri17]
  include_dir = ${CMAKE_INSTALL_PREFIX}/include

Then, issue the following command::

  $ python setup.py install --user

or (if you intend to edit the project)::

  $ python setup.py develop --user

To run the tests with Pytest_::

  $ python -m pytest tests

.. _Breathe: https://breathe.readthedocs.io/
.. _CMake: https://cmake.org/
.. _Doxygen: https://www.doxygen.nl/
.. _Pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
