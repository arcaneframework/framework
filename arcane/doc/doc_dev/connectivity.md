# On-Demand Connectivity Management {#arcanedoc_connectivity}

[TOC]

## Current Status {#arcanedoc_connectivity_current}

Currently, the IItemConnectivity interface allows managing new connectivities,
but it is not designed for incremental updates of elements. By incremental, we
mean that the connectivities must be updated immediately after the addition or
deletion of an entity.

As agreed during a meeting with Stéphane, a new interface
IIncrementalItemConnectivity has been created to manage this type of
connectivity. A generic implementation is available and is called
IncrementalItemConnectivity. It is not optimized but can be applied to any type
of connectivity.

The IIncrementalItemConnectivity interface has several methods divided into
three categories:

- methods identical to IItemConnectivity. These are IItemConnectivity::name(),
  IItemConnectivity::families(), IItemConnectivity::sourceFamily() and
  IItemConnectivity::targetFamily().

- add/remove methods. These are IIncrementalItemConnectivity::addConnectedItem()
  and IIncrementalItemConnectivity::removeConnectedItem(). These two methods
  allow adding or removing an entity. \note For now, these methods take an
  ItemInternal* argument. It would be worth seeing if it wouldn't be better with
  an ItemLocalId.
- notification methods. They are available so that the families associated with
  these connectivities can notify them of an internal modification. There are 4
  methods. IIncrementalItemConnectivity::notifySourceFamilyLocalIdChanged() and
  IIncrementalItemConnectivity::notifyTargetFamilyLocalIdChanged() are called
  when the source or target family is compacted. The method
  IIncrementalItemConnectivity::notifySourceItemAdded() is called when an
  element is added to the source family. This allows, in particular, resizing
  internal arrays. Finally, the last method
  IIncrementalItemConnectivity::notifyReadFromDump() is called after a reread
  following a rollback or recovery.

These notification methods can evolve in several ways:

- we can add the notion of an event to the families, with one event per
  notification type, and then each family registers itself. The advantage of
  this is that it does not require a specific method in the
  IIncrementalItemConnectivity interface, but it makes the code less readable
  (because the registration of families is not easily visible in the source) and
  does not easily allow managing the order of calls between different
  connectivities if needed.
- we can also make these notifications available to connectivities that
  implement IItemConnectivity. In this case, a common base interface with
  IIncrementalItemConnectivity would be useful.

To correctly manage updates following compaction, I had to make the write access
to ItemFamily::m_infos private in ItemFamily. As a result, instead of calling
the entity compaction directly via DynamicMeshKindInfos::beginCompactItems() and
DynamicMeshKindInfos::finishCompactItems(), we must call the corresponding
methods of ItemFamily, which will handle the delegation and notify the
incremental connectivities of this change. \note This is also where we could
notify other connectivities of a potential compaction.

These new features are currently only implemented for the node->face
connectivity of NodeFamily. This connectivity is done in duplicate with the
classic connectivity. In check mode, we verify after every change in the family
that the new connectivity and the old one (which serves as a reference) are the
same. \warning However, currently the new connectivity is never used directly.
Access via ItemInternal is always done with the old mechanism. \note Currently,
there is only one operation that is not implemented with the new features, which
is sorting by increasing uniqueId() in the method
DynamicMesh::_sortInternalReferences().

For this node->face connectivity, I also implemented the current connectivity
via the IIncrementalItemConnectivity interface. The class that manages this is
NodeFaceCompactIncrementalItemConnectivity. As a result, the same mechanism is
used for the old and new connectivity, and it is therefore quite general.

To activate the new connectivity, you must set the environment variable
ARCANE_CONNECTIVITY_POLICY to 1. Eventually, obviously, we will have to do
something else.

Currently, all test base cases work with this new connectivity. I also tested
(briefly) on the integration base of our codes under Arcane and I did not have
any problems.

## Next Phases {#arcanedoc_connectivity_next_phases}

If the proof of concept is okay, the following phases will be (more or less in
this order):

- modify ItemInternal to use the new connectivity if it is defined.
- use the IIncrementalItemConnectivity interface to manage all connectivities,
  even the old ones. This phase can be done in several sub-phases depending on
  the connectivities. The 'classic' connectivities (nodes, edges, faces, and
  meshes) first, the more complicated connectivities (especially AMR)
  afterwards.
- optimize IncrementalItemConnectivity, particularly by managing pre-allocation
  to avoid reallocating every time. We will also need to manage compaction.
- be able to activate the old or the new connectivity via configuration.
