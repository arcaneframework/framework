# Handling parallelism in %Arcane {#arcanedoc_parallel_intro}

[TOC]

<!-- presents how %Arcane handles parallelism through domain partitioning. -->

## Generalities {#arcanedoc_parallel_intro_general}

In ARCANE, parallelism is managed by message passing. The mesh is partitioned
into several subdomains. Each subdomain may be supplemented by one or more
layers of ghost meshes that represent a duplication of entities for which
synchronization will be required. Each processor performs calculations on a
subdomain and regularly synchronizes its variables with the other processors.

Generally, there is no single solution to the synchronization problem:
- for certain variables, it is possible either to perform the calculation over
  the entire subdomain, meaning on both ghost entities and native entities, or
  to perform the calculation only on the native entities and then synchronize
  the calculated values. The choice between one solution or the other depends
  generally on two parameters: the time required to perform the calculation and
  the time required to perform the synchronization. Each of these parameters
  depends itself on other parameters such as processor power, interconnection
  network bandwidth, etc.;
- when the number of ghost entity layers is increased, certain variables that
  are only used temporarily during an iteration no longer need to be
  synchronized. This thus allows reducing the number of synchronizations, but
  conversely, each synchronization will involve more processors and ghost
  entities.

In order not to unnecessarily complicate the handling of parallelism, ARCANE
currently operates using a single layer of ghost meshes and favors calculation
over the number of synchronizations. When developing your numerical modules, it
is advisable to follow the same criterion.

ARCANE has its own parallel partitioning service. This service uses the Métis
algorithm. It is used for balancing the processor load.

## Sequential vs Parallel Calculations {#arcanedoc_parallel_intro_seqvspar}

ARCANE uses the same numbering order in each subdomain. This helps to limit the
number of synchronizations as much as possible. The meshes, nodes, and faces are
always described in the same order regardless of the subdomain and regardless of
the partitioning.

If all operations are performed by iterating over a group of nodes or meshes,
the result is identical in sequential and parallel execution.

## Available Operations {#arcanedoc_parallel_intro_operation}

The operations offered by the parallelism service are as follows:

- barriers,
- sends / receives,
- groupings,
- reductions,
- broadcasts,
- serialization (pack / unpack).

Details of these operations are available in the online documentation for the
`Arcane::IParallelMng` interface.

Note that code variables can directly call the parallelism operations that are
consistent with their type. For example, a variable of type
`Arcane::VariableScalarReal` named `m_density_ratio_maximum` can call the
reduction operation:

```cpp
m_density_ratio_maximum.reduce(Parallel::ReduceMax);
``` 

## Implementation {#arcanedoc_parallel_intro_impl}

Parallelism in ARCANE is designed as a service (see \ref
arcanedoc_core_types_service). All available operations (synchronization,
reduction...) are interfaced by the service. The only implementation of this
service developed to date is MPI.

To run code in parallel with MPI, the procedure depends on the target Mpi
implementation. For example, if you use a version of mpich2, you must:
- launch the `mpd` daemon in an xterm window,
- set the environment variable `ARCANE_PARALLEL_SERVICE` to the value `Mpi`
- execute your application by typing the command:
  `mpiexec -n nb_proc nom_executable`

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_concurrency
</span>
</div>
