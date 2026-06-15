# What is %Arcane? {#arcanedoc_getting_started_about}

[TOC]

%Arcane is a development environment for parallel numerical calculation codes.
It supports the architectural aspects of a calculation code such as data
structures for the mesh, parallelism, but also aspects more related to the
environment, such as dataset configuration.

%Arcane can be used in two ways:

- the framework mode. This is the classic mode in which %Arcane manages
  all the elements of a calculation code. This mode is described in chapter
  \ref arcanedoc_getting_started_mode_full. In this mode, %Arcane automatically
  manages the calculation initialization, the time loop, the post-processing,
  and the safeguards/recovery.
- the standalone mode. In this mode, %Arcane is used as a simple library
  offering functionalities for parallelism, mesh management, or accelerator
  management. This mode is described in chapter
  \ref arcanedoc_execution_direct_execution

## Installation

The installation procedure is
described [here](https://github.com/arcaneframework/framework)

## Public API

The public API of %Arcane contains the following header directories:

```
arcane/utils
arcane/core/*
arcane/geometry
arcane/accelerator/core
arcane/accelerator
arcane/launcher
arcane/materials
arcane/hdf5
arcane/cartesianmesh
```

%Arcane uses the following components of %Arccore. They usually do not need to
be included directly by the user:

```
arccore/base
arccore/collections
arccore/concurrency
arccore/message_passing
arccore/serialize
arccore/trace
```

The other directories are considered internal to %Arcane and should not be used.

## Usage in a Simulation Code {#arcanedoc_getting_started_mode_full}

A simulation code, whatever it may be, can be seen as a system that takes
certain values as input and provides output values by performing *operations*.
Since it is impossible to handle all types of simulation codes, %Arcane is
restricted to calculation codes having the following properties:

- the execution flow can be described as a repetition of a sequence of
  operations, each execution of the sequence of operations being called an
  *iteration*;
- The simulation domain is discretized in space into a set of elements, the
  *mesh*. This mesh can be 1D, 2D, or 3D. A mesh consists of at most four types
  of elements: nodes, edges, faces, and cells. The values manipulated by the
  code are based on one of these element types.

Each of the terms described above has its own terminology in %Arcane:

- a code operation is called
  an [entry point](\ref arcanedoc_core_types_axl_entrypoint).
- The description of the sequence of operations is called
  the [time loop](\ref arcanedoc_core_types_timeloop)
- the manipulated values are
  called [variables](\ref arcanedoc_core_types_axl_variable). For example,
  temperature and pressure are variables.

In general, a simulation code can be decomposed into several distinct parts.
For example, the numerical calculation itself and the part performing outputs
for post-processing. Similarly, a code can use multiple physics: hydrodynamics,
thermal, ...
To ensure the modularity of a code, %Arcane provides what is called
a [module](\ref arcanedoc_core_types_module) which groups all the entry points
and variables corresponding to a given part of the code.

Finally, modules often need to leverage certain functionalities. For example, a
thermal module and a hydrodynamic module may want to use the same numerical
scheme. To ensure code capitalization, %Arcane provides what is called
a [service](\ref arcanedoc_core_types_service)

The 4 concepts described above (entry point, variable, module, and service) are
the basic concepts of %Arcane. They are described in more detail in the document
\ref arcanedoc_core_types. Nevertheless, before seeing in more detail how these
three objects work, one must know the basic concepts presented in the chapters
of this document.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started
</span>
<span class="next_section_button">
\ref arcanedoc_getting_started_basicstruct
</span>
</div>
