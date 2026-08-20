# The Item Family {#arcanedoc_core_types_item_family}

[TOC]

The \arcane{IItemFamily} interface (defined in `arcane/core/IItemFamily.h`) is
the interface of an *entity family*. A family manages all the entities of the
same kind (\arcane{Item::kind}()) and is attached to a mesh
(\arcane{IMesh}, see \ref arcanedoc_core_types_mesh).

For any mesh, there is exactly one family of nodes (\arcane{Node}), one of
edges (\arcane{Edge}), one of faces (\arcane{Face}), and one of cells
(\arcane{Cell}). These entities are called *base mesh entities* and the
associated families are the *base mesh families*.

Depending on the implementation, there may also be families of particles
(\arcane{Particle}), dual nodes, links, or degrees of freedom (\arcane{DoF}).
Depending on the requested connectivity, a family may not have elements. For
example, by default in 3D, edges are not created.

The entity kinds are defined by the \arcane{eItemKind} enumeration:
`IK_Node`, `IK_Edge`, `IK_Face`, `IK_Cell`, `IK_DoF`, `IK_Particle`.

## Getting a Family {#arcanedoc_core_types_item_family_get}

The families of a mesh are retrieved from the \arcane{IMesh} interface:

- \arcane{IMesh::nodeFamily()}, \arcane{IMesh::edgeFamily()},
  \arcane{IMesh::faceFamily()}, \arcane{IMesh::cellFamily()}: the base
  families,
- \arcane{IMesh::itemFamily()}: the family of the base kind \a kind,
- \arcane{IMesh::findItemFamily()}: the family named \a name (throws an
  exception if it does not exist),
- \arcane{IMesh::createItemFamily()}: creates a new family,
- \arcane{IMesh::itemFamilies()}: the collection of all families of the mesh.

A family provides its name (\arcane{IItemFamily::name()}, and
\arcane{IItemFamily::fullName()} which includes the mesh name), its kind
(\arcane{IItemFamily::itemKind()}), and its associated objects:
\arcane{IItemFamily::mesh()}, \arcane{IItemFamily::parallelMng()},
\arcane{IItemFamily::traceMng()}, \arcane{IItemFamily::policyMng()} (the
behaviors/policies manager), and \arcane{IItemFamily::properties()}.

## Entity Identifiers {#arcanedoc_core_types_item_family_ids}

Each entity in a family has two identifiers:

- the *local identifier*, \arcane{Item::localId()}: the position of the entity
  within the family of the local subdomain. This identifier **may be modified**
  when the family evolves (addition, removal, or compaction of entities), and
  the local identifiers are not necessarily contiguous.
  \arcane{IItemFamily::nbItem()} gives the number of entities and
  \arcane{IItemFamily::maxLocalId()} the maximum local identifier (which is the
  size required to dimension variables on the family).
- the *unique identifier*, \arcane{Item::uniqueId()}: stable across the
  computation and unique across all subdomains. It is the identifier to use to
  refer to an entity from one subdomain to another.

By default, a family has a conversion table from \arcane{Item::uniqueId()} to
\arcane{Item::localId()}. This table must exist to allow:
- the guarantee that the unique number is unique within the subdomain and
  across all subdomains,
- the conversion methods \arcane{IItemFamily::itemsUniqueIdToLocalId()},
- the presence of the family entities in multiple subdomains,
- the synchronizations,
- the partial variables on the family.

It is possible to enable or disable this conversion table with
\arcane{IItemFamily::setHasUniqueIdMap()} (and query it with
\arcane{IItemFamily::hasUniqueIdMap()}), but only if no entity has been
created, and not on the node, edge, face, and cell families.

To convert unique numbers to local numbers:

```cpp
IItemFamily* family = mesh->cellFamily();
Int32Array local_ids(nb_unique);
family->itemsUniqueIdToLocalId(local_ids, unique_ids, true);
// local_ids[i] is the local identifier of the entity with unique number
// unique_ids[i], or NULL_ITEM_ID if not found (and a fatal error if
// do_fatal is true).
```

\warning When a family is modified by adding or removing entities, the
variables and groups relying on this family are no longer usable until
\arcane{IItemFamily::endUpdate()} is called, and the entity instances
(\arcane{Item}) are invalidated. To retain a reference to an entity, you must
either use a group (\arcane{ItemGroup}) or keep its unique number and use
\arcane{IItemFamily::itemsUniqueIdToLocalId()}.

## Groups and Variables on the Family {#arcanedoc_core_types_item_family_groups}

A family manages the groups and the variables of its entities:

- \arcane{IItemFamily::allItems()}: the group of all the entities of the
  family,
- \arcane{IItemFamily::groups()}: the collection of groups in the family,
- \arcane{IItemFamily::findGroup()}: searches for a group (by name, optionally
  creating it if it does not exist),
- \arcane{IItemFamily::createGroup()}: creates a group,
- \arcane{IItemFamily::destroyGroups()}: deletes all the groups of the family,
- \arcane{IItemFamily::findVariable()}: searches for the variable named \a name
  associated with the family,
- \arcane{IItemFamily::usedVariables()}: adds the list of variables used by the
  family to a collection.

Variables defined on the entities of the family (such as the AXL-declared
variables of a module on the cell family) are resized automatically when the
family is updated.

## Modifying the Entity List {#arcanedoc_core_types_item_family_modify}

After modifying the entity list (adding or removing entities),
\arcane{IItemFamily::endUpdate()} must be called. It updates the groups and
resizes the variables on the family.

For optimization purposes, when performing several successive modifications,
it is possible to use the partial update methods (reserved for experienced
users):

- \arcane{IItemFamily::partialEndUpdate()}: updates the internal structures
  without updating the groups or variables. Only the `allItems()` group is
  available.
- \arcane{IItemFamily::partialEndUpdateGroup()}: updates a single group after
  the modification (removes the entities that were destroyed).
- \arcane{IItemFamily::partialEndUpdateVariable()}: resizes a single variable
  after the modification.

Other modification-related methods:

- \arcane{IItemFamily::clearItems()}: deletes all the entities of the family
  (be careful not to destroy entities used by another family; it is generally
  safer to use \arcane{IMeshModifier::clearItems()} to delete all the elements
  of the mesh),
- \arcane{IItemFamily::compactItems()}: compresses the entities so that the
  local identifiers are renumbered from 0 to `nbItem()-1`,
- \arcane{IItemFamily::setItemSortFunction()} /
  \arcane{IItemFamily::itemSortFunction()}: sets the entity sorting function
  (by default, entities are sorted by ascending unique number),
- \arcane{IItemFamily::notifyItemsOwnerChanged()}: notifies that the entities
  owned by the subdomain have changed,
- \arcane{IItemFamily::notifyItemsUniqueIdChanged()}: notifies that the unique
  identifiers of the entities have changed,
- \arcane{IItemFamily::addGhostItems()}: allocates ghost entities (the entities
  present on another subdomain); \arcane{IItemFamily::endUpdate()} must be
  called after the allocations,
- \arcane{IItemFamily::copyItemsValues()} /
  \arcane{IItemFamily::copyItemsMeanValues()}: copies (or averages) the values
  of the entities of a list into the entities of another list,
- \arcane{IItemFamily::experimentalChangeUniqueId()}: changes the unique number
  of an entity (experimental: only for entities not yet connected to others).

## Synchronization and Ghost Entities {#arcanedoc_core_types_item_family_sync}

In parallel, the entities of a family can be shared by several subdomains
(ghost entities). The synchronization of the variables between the subdomains
is managed by the family:

- \arcane{IItemFamily::computeSynchronizeInfos()}: constructs the structures
  necessary for the synchronization. This collective operation must be
  performed every time the entities change ownership (for example, during a
  load balancing),
- \arcane{IItemFamily::synchronize()}: synchronizes the given variables (which
  must come from this family and must not be partial) on all the entities of
  the family, or on a list of entities,
- \arcane{IItemFamily::allItemsSynchronizer()}: the synchronizer on all the
  entities of the family,
- \arcane{IItemFamily::reduceFromGhostItems()}: applies a reduction operation
  from the ghost items (the inverse of a synchronization),
- \arcane{IItemFamily::getCommunicatingSubDomains()}: the list of subdomains
  communicating for the entities,
- \arcane{IItemFamily::itemsNewOwner()}: the variable containing the number of
  the new owning subdomain of the entities (used for mesh partitioning).

See \ref arcanedoc_parallel_intro for a general presentation of the
synchronization in %Arcane.

## Adjacencies and Connectivity {#arcanedoc_core_types_item_family_adjacency}

The family provides methods to compute the adjacency lists:

- \arcane{IItemFamily::findAdjacencyItems()}: searches for the list of
  entities of a given sub-kind, linked by a given link kind, over a number of
  layers. If the list does not exist, it is created.
- \arcane{IItemFamily::localConnectivityInfos()} /
  \arcane{IItemFamily::globalConnectivityInfos()}: the information on the
  connectivity, local to the subdomain or global across all subdomains.

The connectivity between the entities of the mesh is described by the
connectivity mechanism presented in \ref arcanedoc_entities_connectivity_internal.

## Sub-mesh Nesting {#arcanedoc_core_types_item_family_submesh}

A sub-mesh (see \ref arcanedoc_core_types_mesh_primary) has families nested
with respect to the families of its parent mesh:

- \arcane{IItemFamily::parentFamily()}: the parent family (or \c nullptr),
- \arcane{IItemFamily::childFamilies()}: the child families,
- \arcane{IItemFamily::parentFamilyDepth()}: the nesting depth of the current
  family,
- \arcane{IItemFamily::setParentFamily()} /
  \arcane{IItemFamily::addChildFamily()}: set up the nesting (to be used before
  \arcane{IItemFamily::build()} for dynamically constructed sub-meshes).

## Other Features {#arcanedoc_core_types_item_family_other}

- \arcane{IItemFamily::view()}: a view on the entities of the family (valid
  only as long as the family does not evolve),
- \arcane{IItemFamily::toParticleFamily()} /
  \arcane{IItemFamily::toDoFFamily()}: convert the family to the
  \arcane{IParticleFamily} or \arcane{IDoFFamily} interface when the kind
  matches,
- \arcane{IItemFamily::checkValid()} /
  \arcane{IItemFamily::checkValidConnectivity()}: check the validity of the
  internal structures,
- \arcane{IItemFamily::checkUniqueIds()}: collectively checks that the given
  unique identifiers are truly unique for all subdomains,
- \arcane{IItemFamily::itemListChangedEvent()}: an event notified when
  entities are added or removed,
- \arcane{IItemFamily::prepareForDump()} /
  \arcane{IItemFamily::readFromDump()}: prepare or read the family data for a
  dump.

## Key Points {#arcanedoc_core_types_item_family_notes}

- A family is the per-mesh container of all the entities of one kind: it owns
  their numbering, their groups, their variables, and their synchronization.
- The \arcane{Item::localId()} of an entity can change when the family evolves;
  the \arcane{Item::uniqueId()} is stable. Use the unique number (and
  \arcane{IItemFamily::itemsUniqueIdToLocalId()}) to refer to an entity across
  modifications or across subdomains.
- After any modification of the entity list, call
  \arcane{IItemFamily::endUpdate()} (or the partial update methods) before
  using the groups and variables again.
- For base mesh families (node, edge, face, cell), the unique-to-local
  conversion table is mandatory; for other families it can be disabled before
  any entity is created.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_mesh
</span>
</div>
