############
Installation
############


Installing and testing the library
==================================

bri17 has the following dependencies, required for the tests only

- `Eigen <http://eigen.tuxfamily.org/>`_
- `FFTW <http://www.fftw.org/>`_

After cloning the repository, ``cd`` into the root directory of the project::

  $ mkdir build
  $ cd build
  $ cmake -G Ninja -D CMAKE_INSTALL_PREFIX="C:\opt\bri17" ..
  $ ninja install

(you can tweak the value of ``CMAKE_INSTALL_PREFIX`` to your liking). To run the
tests::

  $ ninja test


Compiling your first program
============================

In this section, we create a simple example that instantiates a
:cpp:class:`bri17::CartesianGrid` and prints it to ``stdout``. The layout of
this project is::

  example/
    ├─ src/
    │   └─ bri17_example.cc
    └─ CMakeLists.txt

And the files are listed below.

:download:`bri17_example.cc <example/src/bri17_example.cc>`

.. literalinclude:: example/src/bri17_example.cc

:download:`CMakeLists.txt <example/CMakeLists.txt>`

.. literalinclude:: example/CMakeLists.txt
   :language: cmake

Compilation follows the standard procedure::

  $ cd example
  $ mkdir build
  $ cd build
  $ cmake -G Ninja -D bri17_DIR="C:\opt\bri17\share\bri17\cmake\bri17" ..
  $ ninja

If bri17 was installed in a standard location, you might not need to specify the
variable ``bri17_DIR``. If necessary, specify the full path to the folder that
holds the files ``bri17-config.cmake``, ``bri17-config-version.cmake`` and
``bri17-targets.cmake``.

Running the generated program produces the following output::

  $ ./bri17_example.exe
  CartesianGrid<3>={L=[1,2,3,],N=[3,4,5,]}


Building the documentation
==========================

Requires `Doxygen <http://www.doxygen.nl/>`_ and the `Breathe
<https://github.com/michaeljones/breathe>`_ extension to `Sphinx
<https://www.sphinx-doc.org/>`_. ``cd`` into the ``sphinx/`` subdirectory of the
project and issue the following command: ``sphinx-build . ../docs``. The HTML
docs are located in the ``$PROJECT_ROOT/docs`` directory.
