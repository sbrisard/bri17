############
Installation
############

Compile your first program
==========================

In this section, we create a simple example that instantiates a
:cpp:class:`bri17::CartesianGrid` and prints it to ``stdout``. The layout of this project is::

  example/
    ├─ src/
    │   └─ bri17_example.cc
    └─ CMakeLists.txt

The contents of the :download:`bri17_example.cc <example/src/bri17_example.cc>`
file is

.. literalinclude:: example/src/bri17_example.cc

And the :download:`CMakeLists.txt <example/CMakeLists.txt>`

.. literalinclude:: example/CMakeLists.txt
   :language: cmake

.. todo:: Is it possible to get rid of the ``find_package(Eigen3 3.3 REQUIRED
          NO_MODULE)``?

Documentation
=============

Requires `Doxygen <http://www.doxygen.nl/>`_ and the `Breathe
<https://github.com/michaeljones/breathe>`_ extension to `Sphinx
<https://www.sphinx-doc.org/>`_. In the console, issue the following commands::

  cd $PROJECT_ROOT
  sphinx-build . ../docs

The HTML docs are located in the ``$PROJECT_ROOT/docs`` directory.
