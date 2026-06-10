# Load balancing on the mesh {#arcanedoc_parallel_loadbalance}

[TOC]

## Introduction {#arcanedoc_parallel_loadbalance_introduction}

%Arcane has a load balancing mechanism by redistributing mesh elements between
subdomains. This mechanism manages the exchange of mesh entities as well as
associated variables. It is therefore largely transparent to the user.

Load balancing management is done via two interfaces:

- \arcane{ICriteriaLoadBalanceMng} which allows specifying the criteria to
  consider for load calculation.
- \arcane{IMeshPartitioner} which allows determining the entities that must
  migrate and performing the migration.

The class \arcane{MeshCriteriaLoadBalanceMng} implementing the
\arcane{ICriteriaLoadBalanceMng} interface allows, during initialization,
specifying one or more mesh variables that will contain the weight of each mesh
element for load calculation.

\deprecated Using `ISubDomain::loadBalance()` to define criteria is now
obsolete.

For example:
```cpp
Arcane::VariableCellReal cells_weight(...);
Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
mesh_criteria.addCriterion(cells_weight);
```

\remark The `mesh_criteria` object can be destroyed without problems after use.
The registered variables will still exist after its destruction.

\warning Calling the Arcane::MeshCriteriaLoadBalanceMng::reset() method will
affect all criteria added since the beginning (for a given mesh).
Example:
```cpp
Arcane::VariableCellReal cells_weight(...);
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
  mesh_criteria.addCriterion(cells_weight);
}
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
  mesh_criteria.reset(); // Here, the criterion represented by the variable "cells_weight" is also removed.
}
```

The weight calculation is the responsibility of the user code. The partitioner
will then redistribute the mesh, attempting to best balance the weights across
all subdomains. For example, if a costly method is called a different number of
times for each mesh element, it is possible to fill *cells_weight* with the
number of calls made.

In general, after a repartitioning, these variables used as criteria must be
reset to zero.

Repartitioning and balancing are done via the IMeshPartitioner interface
service. It is possible to obtain an instance of this service by specifying the
following line in the 'axl' file:

```xml
<service-instance
 name    = "partitioner"
 type    = "Arcane::IMeshPartitioner"
 default = "DefaultPartitioner"
/>
```

In this case, the partitioner will be accessible via the following method:

```cpp
options()->partitioner()
```

To schedule a repartitioning during the calculation, you must call
ITimeLoopMng::registerActionMeshPartition() specifying the desired partitioner.
The repartitioning will be performed at the end of the current iteration. In a
module, you can do this:

```cpp
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner());
```

\remark This call is only valid for one time step. If you want to repartition at
every time step (which can be quite costly in terms of computation), it is
necessary to register the partitioner at every time step.

Repartitioning performs the transfer of all mesh entities and associated
variables. If the user code needs to perform other operations after a balancing,
it is possible to specify an entry point for this. In the time loop, entry
points with the attribute 'where="on-mesh-changed"' are called after a
balancing. For example:

```xml
<time-loop name="LoadBalanceLoop">
 <modules>...</modules>
 <entry-points where="on-mesh-changed">
  <entry-point name="MyModule.OnMeshChanged"/>
 </entry-points>
</time-loop>
```

\note Currently (March 2017), load balancing only works with a single layer of
ghost meshes.

## Multi-mesh {#arcanedoc_parallel_loadbalance_multimesh}

Since %Arcane handles multi-meshing, it is also possible to balance the load of
several meshes.

The balancing is independent for each mesh (in the future, it will be possible
to define criteria to balance multiple meshes that require "common" balancing).

Let's take two meshes:

```cpp
IMesh* mesh0 = subDomain().meshes()[0];
IMesh* mesh1 = subDomain().meshes()[1];
```

And let's take the previous example but with two meshes:

```cpp
Arcane::VariableCellReal cells_weight_mesh0(...);
Arcane::VariableCellReal cells_weight_mesh1(...);
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh0->handle());
  mesh_criteria.addCriterion(cells_weight_mesh0);
}
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh1->handle());
  mesh_criteria.addCriterion(cells_weight_mesh1);
}
```

\note To create a variable for the second mesh, you can do:
```cpp
Arcane::VariableCellReal cells_weight_mesh1(VariableBuildInfo(mesh1->handle(), "CellsWeight"))
```

Regarding the partitioner service instances, it is possible to specify a mesh in
the axl via the `mesh-name` attribute:

```axl
<service-instance
 name      = "partitioner0"
 type      = "Arcane::IMeshPartitioner"
 default   = "DefaultPartitioner"
 mesh-name = "Mesh0"
/>

<service-instance
 name      = "partitioner1"
 type      = "Arcane::IMeshPartitioner"
 default   = "DefaultPartitioner"
 mesh-name = "Mesh1"
/>
```

Finally, to schedule repartitioning during the calculation, it is possible to
do:

```cpp
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner0());
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner1());
```

\note It is also possible to create the partitioner service instances in the
code and schedule their calls during the calculation:
```cpp
Ref<IMeshPartitionerBase> partitioner0 = ServiceBuilder<IMeshPartitionerBase>::createReference(subDomain(), "DefaultPartitioner", mesh0);
Ref<IMeshPartitionerBase> partitioner1 = ServiceBuilder<IMeshPartitionerBase>::createReference(subDomain(), "DefaultPartitioner", mesh1);
...
subDomain()->timeLoopMng()->registerActionMeshPartition(partitioner0.get());
subDomain()->timeLoopMng()->registerActionMeshPartition(partitioner1.get());
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_simd
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_shmem
</span>
</div>
