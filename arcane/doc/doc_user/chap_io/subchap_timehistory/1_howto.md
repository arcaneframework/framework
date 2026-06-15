# How it works {#arcanedoc_io_timehistory_howto}

[TOC]

The use of TimeHistory is extremely simple.

At each iteration, a value will be recorded for each value history.
If there is no explicit value recording during an iteration, a 0 will be
recorded.
If there are two values recorded during the same iteration for a value history,
the last value recorded will be the one that is actually saved.

Historically, value histories were managed only by process 0.
Today, this is no longer the case. Each process can have its own history, using
the same key.
Furthermore, a history can be linked to a mesh. Thus, we can have a history per
mesh, always using the same key.

\warning To enable multi-process recording, it is necessary to define the
environment variable `ARCANE_ENABLE_NON_IO_MASTER_CURVES=1`.

## GlobalTimeHistoryAdder

The first structure allowing the management of value histories is the
`GlobalTimeHistoryAdder`.
It allows adding values to a history.
"Global" means that the internal variables used to manage this history are
global, linked to the sub-domains.

Suppose we have a mesh shared across four sub-domains (`SD0`, `SD1`, `SD2`,
`SD3`).
Each cell has a pressure. We want to have, for each iteration, the average
pressure of each sub-domain. And in addition, we want to have the average
pressure of the entire domain.

Let's use `avg_pressure` as the key:
\image html avg_pressure.svg

Each sub-domain has an average pressure and there is a global average pressure.
The image presents a single iteration: iteration 0.

To obtain a history like this, we can do this:
\snippet{c++} TimeHistoryAdderTestModule.cc snippet_timehistory_example1

\remark Internally, `GlobalTimeHistoryAdder` uses the internal part of the
`ITimeHistoryMng` passed as a parameter.
The `GlobalTimeHistoryAdder` object can therefore be destroyed without problems.

\note To use `GlobalTimeHistoryAdder`, do not forget to import the necessary
headers:
```cpp
#include <arcane/core/ITimeHistoryMng.h>
#include <arcane/core/GlobalTimeHistoryAdder.h>
```

This piece of code, if called at every iteration, allows obtaining the averages
at each iteration.

## MeshTimeHistoryAdder

The second structure allowing the management of value histories is the
`MeshTimeHistoryAdder`.
Like the first structure, it allows adding values to a history.
"Mesh" means that the internal variables used to manage this history are linked
to the desired mesh. Therefore, each mesh can have a different variable with the
same name.

Let's take the example above but with two meshes.
These two meshes are distributed across four sub-domains. We want to have, for
each sub-domain, the average pressure of the cells of each mesh.
But we still want, as above, the average pressure of each sub-domain.

Let's use the same key: `avg_pressure`:
\image html avg_pressure2.svg

We can see that in addition to the `avg_pressure` of each sub-domain and the
global one, there are `avg_pressure` for the two meshes.

Here is a code example to perform this calculation:
\snippet{c++} TimeHistoryAdderTestModule.cc snippet_timehistory_example2

The difference here is that we iterate over the meshes. To create the
`MeshTimeHistoryAdder`, in addition to an `ITimeHistoryMng*`, we must provide a
mesh handle. This allows linking the history to the mesh.

\remark Internally, `MeshTimeHistoryAdder` uses the internal part of the
`ITimeHistoryMng` passed as a parameter.
The `MeshTimeHistoryAdder` object can therefore be destroyed without problems.

\note To use `MeshTimeHistoryAdder`, do not forget to import the necessary
headers:
```cpp
#include <arcane/core/ITimeHistoryMng.h>
#include <arcane/core/MeshTimeHistoryAdder.h>
```

\note The TimeHistoryMng manages checkpoints, so the user does not have to worry
about it.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_io_timehistory
</span>
<span class="next_section_button">
\ref arcanedoc_io_timehistory_results_usage
</span>
</div>
