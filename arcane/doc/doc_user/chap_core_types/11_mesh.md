# The Mesh {#arcanedoc_core_types_mesh}

[TOC]

The \arcane{IMesh} interface (defined in `arcane/core/IMesh.h`) is the
interface of a mesh. A mesh is composed of a set of *entities* (\arcane{Item}),
such as nodes (\arcane{Node}), edges (\arcane{Edge}), faces (\arcane{Face}), or
cells (\arcane{Cell}). It is also possible to associate other kinds of
entities, such as particles (\arcane{Particle}) or degrees of freedom
(\arcane{DoF}).

Entities of the same kind are managed in a *family* (\arcane{IItemFamily}, see
\ref arcanedoc_core_types_item_family). A mesh has a dimension, retrievable via
\arcane{IMesh::dimension()}, which can be 1, 2, or 3. The dimension
corresponds to the dimension of the cell elements.

When an %Arcane application runs in parallel, the mesh is partitioned into
several subdomains: each subdomain owns a portion of the mesh, completed by one
or more layers of *ghost entities* owned by the other subdomains (see
\ref arcanedoc_parallel_intro). The \arcane{IMesh} instance of a subdomain
represents this local view of the mesh.

## Primary and Secondary Meshes {#arcanedoc_core_types_mesh_primary}

There are two types of meshes implementing \arcane{IMesh}:

- *primary meshes*, which also implement the \arcane{IPrimaryMesh} interface
  and which can be created dynamically or by reading a file. A primary mesh is
  created either when reading the data set, or by programming, by calling one
  of the \arcane{IMainFactory::createMesh()} methods. It is not possible to
  delete a primary mesh.
- *secondary meshes* (also called *sub-meshes*), which depend on a primary
  mesh and represent a subset of it. A sub-mesh is attached to its parent via
  \arcane{IMesh::defineParentForBuild()}, and the parent/child relationship can
  be retrieved with \arcane{IMesh::parentMesh()} / \arcane{IMesh::parentGroup()}
  and \arcane{IMesh::childMeshes()}.

\arcane{IMesh::isPrimaryMesh()} indicates if the instance is a primary mesh,
and \arcane{IMesh::toPrimaryMesh()} returns the instance in the form of an
\arcane{IPrimaryMesh} (throwing an exception if it is not one).

## Getting a Mesh {#arcanedoc_core_types_mesh_get}

The default mesh of a subdomain is retrieved with
\arcane{ISubDomain::defaultMesh()}. It is generally preferable to use
\arcane{ISubDomain::defaultMeshHandle()}, which returns a \arcane{MeshHandle}:
a stable reference to the mesh that always exists, even if the associated mesh
has not yet been created.

A subdomain can have several meshes. They are managed by the mesh manager,
retrieved with \arcane{ISubDomain::meshMng()}, and the list of meshes is
available via \arcane{ISubDomain::meshes()}.

## Mesh Entities and Families {#arcanedoc_core_types_mesh_families}

For any mesh, there is exactly one family of nodes, one of edges, one of faces,
and one of cells. These entities are called *base mesh entities* and the
associated families are the *base mesh families*:

- \arcane{IMesh::nodeFamily()}, \arcane{IMesh::edgeFamily()},
  \arcane{IMesh::faceFamily()}, \arcane{IMesh::cellFamily()}

Depending on the implementation and the requested connectivity, a family may
not contain any element. For example, by default in 3D, edges are not created.

It is also possible to create additional families for other entity kinds or
named families:

```cpp
IMesh* mesh = subDomain()->defaultMesh();
IItemFamily* particles = mesh->createItemFamily(IK_Particle, "MyParticles");
IItemFamily* family = mesh->findItemFamily(IK_Particle, "MyParticles");
```

The number of entities of each base kind is given by \arcane{IMesh::nbNode()},
\arcane{IMesh::nbEdge()}, \arcane{IMesh::nbFace()}, and \arcane{IMesh::nbCell()}
(or \arcane{IMesh::nbItem()} for a generic kind).

The cell types follow the numbering convention of the VTK library (see
\ref arcanedoc_entities_itemtype).

## Groups of Entities {#arcanedoc_core_types_mesh_groups}

An *entity group* (\arcane{ItemGroup}) contains a set of entities of a given
family. Groups are the main way to iterate over the entities of a mesh. The
\arcane{IMesh} interface provides the following built-in groups:

| Group | Description |
|---|---|
| \arcane{IMesh::allNodes()} / \arcane{IMesh::allEdges()} / \arcane{IMesh::allFaces()} / \arcane{IMesh::allCells()} | All entities of the subdomain (including ghosts) |
| \arcane{IMesh::ownNodes()} / \arcane{IMesh::ownEdges()} / \arcane{IMesh::ownFaces()} / \arcane{IMesh::ownCells()} | Entities owned by the subdomain |
| \arcane{IMesh::outerFaces()} | Faces on the boundary of the subdomain |

Other groups can be created and searched by name:

- \arcane{IMesh::groups()}: the collection of all groups,
- \arcane{IMesh::findGroup()}: the group named \a name, or a null group if none
  exists,
- \arcane{IMesh::destroyGroups()}: destroys all groups.

Groups are the recommended way to *retain* a set of entities across mesh
modifications: entity instances (\arcane{Item}) and their local identifiers
may become invalid after a mesh update, while groups are updated
automatically.

## Modifying a Mesh {#arcanedoc_core_types_mesh_modify}

A mesh can be *dynamic*, i.e. able to evolve during the computation
(\arcane{IMesh::isDynamic()}). Modifications are performed through the
\arcane{IMeshModifier} interface, retrieved with \arcane{IMesh::modifier()}:

```cpp
IMesh* mesh = subDomain()->defaultMesh();
IMeshModifier* mod = mesh->modifier();

mod->addCells(0, cells_infos, true);  // add cells described by unique numbers
mod->removeCells(cells_to_remove);    // remove cells (by local identifier)
mod->endUpdate();                     // collective: finalize the modification
```

Once the modifications are made, \arcane{IMeshModifier::endUpdate()} must be
called. This method is collective and, depending on the mesh properties (see
below), sorts and/or compacts the entities, updates the groups, resizes the
variables, and updates the ghost layers.

\warning Sorting or compacting the entities modifies the \arcane{Item::localId()}
of all geometric entities of the mesh. Therefore, you must not retain an entity
via the \arcane{Item} class (or a derived class) or its local identifier after
calling this method. Use groups (or unique numbers) instead.

## Creating a Primary Mesh by Programming {#arcanedoc_core_types_mesh_create}

To create an empty 2D mesh named `"Mesh2"`:

```cpp
// sd is the current subdomain.
ISubDomain* sd = ...;
IParallelMng* pm = sd->parallelMng();
IMainFactory* mf = sd->application()->mainFactory();
IPrimaryMesh* new_mesh = mf->createMesh(sd, pm, "Mesh2");
new_mesh->setDimension(2);
new_mesh->allocateCells(0, Int64ConstArrayView(), false);
new_mesh->endAllocate();
```

The created mesh exists across all ranks of the \arcane{IParallelMng} passed
as an argument. To create a mesh on a single processor, you can use
\arcane{IParallelMng::sequentialParallelMng()}. The mesh dimension must be set
with \arcane{IPrimaryMesh::setDimension()} before allocating cells.

It is also possible to read a mesh directly from a file using an
implementation of \arcane{IMeshReader}:

```cpp
IMeshReader* reader = ServiceBuilder<IMeshReader>::createInstance(sd, "VtkLegacyMeshReader");
reader->readMeshFromFile(new_mesh, XmlNode(), "sod.vtk", "/tmp", false);
delete reader;
```

## Mesh Properties {#arcanedoc_core_types_mesh_properties}

A mesh has properties, accessible via \arcane{IMesh::properties()}:

- \b "sort" (boolean, true by default): entities must be sorted by increasing
  unique number after a mesh modification. This ensures that operations always
  happen in the same parallel order, regardless of the number of subdomains.
- \b "compact" (boolean, true by default): entities must be compacted after a
  mesh modification, so that there are no gaps in the local numbering
  (the local identifiers are renumbered from 0 to the number of entities).
  Compacting is a costly operation because it requires updating all variables.
- \b "dump" (boolean, true by default): the mesh must be saved during a
  checkpoint.
- \b "edges" (boolean, false by default): edges must be created in a 3D mesh.
- \b "sort-subitemitem-group" (boolean, false by default): whether dynamically
  created groups of connected entities (e.g. `allCells().nodes()`) are sorted.
  Must be set during mesh creation.

## Load Balancing and Redistribution {#arcanedoc_core_types_mesh_loadbalance}

When load balancing redistributes the mesh between processors, the
\arcane{IPrimaryMesh::itemsNewOwner()} variable of the primary mesh contains,
for each entity, the number of its new owning subdomain. These variables must
be synchronized, and the redistribution is then performed with
\arcane{IPrimaryMesh::exchangeItems()}:

- the subdomain gives the cells that now belong to other subdomains to them,
  and receives the new cells (and the same for nodes, edges, and faces),
- the values of the variables and of the entity groups are exchanged,
- the method triggers an implicit call to \arcane{IMeshModifier::endUpdate()}.

After the call, you may need to execute the mesh change entry points
(\arcane{ITimeLoopMng::execOnMeshChangedEntryPoints()}).

## Other Features {#arcanedoc_core_types_mesh_other}

- **AMR**: if adaptive mesh refinement is activated
  (\arcane{IMesh::isAmrActivated()}), the mesh provides the groups of active
  cells (\arcane{IMesh::allActiveCells()}, \arcane{IMesh::ownActiveCells()})
  and of cells at a given level (\arcane{IMesh::allLevelCells()}). See
  \ref arcanedoc_entities_amr_cartesianmesh.
- **Ghost layers**: \arcane{IMesh::ghostLayerMng()} gives access to the ghost
  layer manager and \arcane{IMesh::updateGhostLayers()} rebuilds the ghost
  layers.
- **Connectivity**: \arcane{IMesh::connectivity()} returns the connectivity
  descriptor (a \arcane{VariableScalarInteger} using the %Arcane connectivity
  numbering, see \ref arcanedoc_entities_connectivity_internal), and
  \arcane{IMesh::indexedConnectivityMng()} the indexed incremental connectivity
  manager.
- **Unique IDs**: \arcane{IMesh::meshUniqueIdMng()} manages the unique
  numbering of the entities.
- **Semi-conforming interfaces**: \arcane{IMesh::computeTiedInterfaces()},
  \arcane{IMesh::hasTiedInterface()}, and \arcane{IMesh::tiedInterfaces()}
  manage the tied interfaces between non-conforming sub-meshes.
- **Partitioning constraints**: \arcane{IMesh::partitionConstraintMng()}
  manages the constraints on the mesh partitioning.
- **Utilities and checks**: \arcane{IMesh::utilities()} gives access to mesh
  utility functions (\arcane{IMeshUtilities}), \arcane{IMesh::checker()} to the
  mesh validity checker, \arcane{IMesh::checkValidMesh()} (local) and
  \arcane{IMesh::checkValidMeshFull()} (collective, across all subdomains) to
  validate the internal structures.
- **Events**: \arcane{IMesh::eventObservable()} provides observables for mesh
  events.
- **User data**: \arcane{IMesh::userDataList()} allows associating user data
  with the mesh.
- **Family network**: \arcane{IMesh::itemFamilyNetwork()} gives access to the
  graph of the connected families.

## Key Points {#arcanedoc_core_types_mesh_notes}

- \arcane{IMesh} is the local (per-subdomain) view of a mesh: it contains the
  entities owned by the subdomain plus the ghost entities.
- Use \arcane{ISubDomain::defaultMeshHandle()} (a \arcane{MeshHandle}) rather
  than \arcane{ISubDomain::defaultMesh()} when the mesh may not have been
  created yet.
- Primary meshes are created by the data set or via
  \arcane{IMainFactory::createMesh()}; sub-meshes are subsets of a primary
  mesh.
- Modify the mesh only through \arcane{IMesh::modifier()}, and always call
  \arcane{IMeshModifier::endUpdate()} after a modification.
- Do not retain \arcane{Item} instances or \arcane{Item::localId()} values
  across an \arcane{IMeshModifier::endUpdate()}: use groups or unique numbers.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_subdomain
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_item_family
</span>
</div>
