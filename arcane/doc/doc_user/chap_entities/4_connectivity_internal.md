# Entity Connectivity Management {#arcanedoc_entities_connectivity_internal}

[TOC]

This page groups information on the developments made in %Arcane for managing
new connectivities. It is based on the CEA version from February 2017, which
corresponds to %Arcane versions 2.5.0 and later.

To meet new needs, the entity connectivity management mechanism in %Arcane
evolved starting in 2017.

The historical mechanism's primary goal was to save memory and stored all of an
entity's connectivities consecutively in memory. However, this presents two
drawbacks:
- this mechanism is not easily extensible if you want to add new connectivities
  or if you do not want to use certain connectivities. Initially, only nodes,
  faces, and cells were managed. Today, there are dual nodes, links, degrees of
  freedom, AMR, and all these connectivities complicate management.
- it is more difficult to benefit from memory cache effects when traversing only
  one type of connectivity.

The new mechanism allows for the complete separation of each connectivity type
and potentially the specialization of a connectivity type based on certain
needs (for example, depending on the mesh type).

It resolves the two previous drawbacks, but in return, it results in an increase
in memory usage for unstructured meshes. With the old mechanism, 1 index (of
type Arcane::Int32) was sufficient for each entity to access connectivity
information, whereas with the new mechanism, 2 indices (position + number of
connectivities) are required per connectivity. For example, in the case of
classic mesh entities (node, edge, face, or cell), 8 indices are therefore
required instead of 1.

The historical mechanism allows access to connectivity information directly via
the entity. For example, to access the 4th node of a cell:
```cpp
Arcane::Cell cell;
Arcane::Node node = cell.node(3);
```

In the long term, access to connectivities might be available in another form,
but for now, this mechanism must be usable to avoid making all current codes
incompatible.

To transition between the old and new connectivity management, compatibility
mechanisms have been implemented. The goal of these mechanisms is to ensure
compatibility at the source level of codes using %Arcane: these codes must be
able to compile without modification with %Arcane versions that integrate the
new connectivities.

To ensure this compatibility, the access mechanism for methods such as
Arcane::Cell::node() is modified and now uses an object of type
Arcane::ItemInternalConnectivityList. The Arcane::ItemInternal class contains a
field Arcane::ItemInternal::m_connectivity which is a pointer to an
Arcane::ItemInternalConnectivityList. All entities of the same family point to
the same value, which is Arcane::ItemFamily::m_item_connectivity_list.

\note Placing this pointer in each Arcane::ItemInternal avoids an indirection
when accessing connectivities, but it is also possible to place it in
Arcane::ItemSharedInfo because it is common to all entities in a family. This
allows for less memory usage for Arcane::ItemInternal (16 bytes instead of 24)
at the cost of an additional indirection. In my tests at CEA, I did not observe
any performance differences between the two mechanisms.

In order not to modify the existing API, the ItemInternal methods for accessing
connectivity have not been modified, and new ones have been added. They use the
V2 suffix. For example, Arcane::ItemInternal::nodesV2() instead of
Arcane::ItemInternal::nodes().

The macro **ARCANE_USE_LEGACY_ITEMINTERNAL_CONNECTIVITY** allows you to choose
at compile time whether the accessors via Arcane::Item, Arcane::Edge,
Arcane::Face, Arcane::Cell, ... use the old or new mechanisms. If this macro is
defined, then:

```cpp
// historical version if macro defined
NodeVectorView Cell::node() { return m_internal->nodes(); }
// new version if macro not defined
NodeVectorView Cell::node() { return m_internal->nodesV2(); }
```

This macro is defined only if %Arcane is compiled with the
**--with-legacy-connectivity** option in the configure script. If this option is
active, it is impossible to access the new connectivities via Item. The only
purpose of this option is to check if the new mechanism contains bugs and to
validate new %Arcane versions on old user codes. We will assume later that this
option is not used and therefore that connectivities are accessed via the V2
methods. To do this, you must use the configuration option
**--without-legacy-connectiviy** in the configure script.

When using the V2 methods, access to each connectivity is done via the
Arcane::ItemInternalConnectivityList class. For each connectivity, there are
three arrays:
- nb_item: number of connected entities
- list: array of localId() of connected entities.
- index: index in \a list of the first connected entity.

The list of entities connected to an entity is stored consecutively in memory,
so the index in the first array allows the others to be retrieved.
\a nb_item and \a index are indexed by the localId() of the entity whose
connectivities you want to access. For example:

```cpp
using namespace Arcane;
Item my_item = ...;
Int32 lid = my_item.localId();
Int32ConstArrayView nb_items = ...;
Int32ConstArrayView list = ...;
Int32ConstArrayView index = ...;
// Number of entities connected to my_item.
Int32 n = nb_items[lid];
// localId() of the first entity connected to my_item.
Int32 c0 = list[ index[lid] ];
```

All connectivities of classic entities (Arcane::Node, Arcane::Edge,
Arcane::Face, and Arcane::Cell) now use this type of access. It is possible to
choose at runtime whether the access is done via historical connectivities or
new connectivities. This is done via the environment variable
**ARCANE_CONNECTIVITY_POLICY**, which allows associating one of the enumerated
values #Arcane::InternalConnectivityPolicy. This association is made in the
DynamicMesh constructor. The Arcane::IMesh::_connectivityPolicy() method allows
retrieving the chosen policy. There are currently 4 possible values:
- Arcane::InternalConnectivityPolicy::Legacy: indicates that only historical
  connectivities are allocated and therefore that ItemInternal::m_connectivity
  is not used. It is therefore not possible to access the new connectivity
  mechanisms in this mode. This mode corresponds most closely to older versions
  of Arcane, particularly regarding memory usage.
- Arcane::InternalConnectivityPolicy::LegacyAndAllocAccessor: indicates that
  only historical connectivities are allocated and that
  Arcane::ItemInternal::m_connectivity uses the connectivities defined in
  Arcane::ItemFamily::m_items_data. The boolean
  Arcane::ItemFamily::m_use_legacy_connectivity_policy
  is true in this case. This mode allocates connectivity accessors via
  ItemInternal::m_connectivity and therefore uses more memory than the
  InternalConnectivityPolicy::Legacy mode.
- Arcane::InternalConnectivityPolicy::LegacyAndNew: is identical to the
  InternalConnectivityPolicy::LegacyAndAllocAccessor value but additionally the
  new connectivities are allocated. However, they are not used by the Item and
  Internal classes. In check mode, the values of the old and new connectivities
  are verified upon every mesh modification. This mode therefore allows for
  validating the new mechanisms. For this mode,
  Arcane::ItemFamily::m_use_legacy_connectivity_policy is also true.
- Arcane::InternalConnectivityPolicy::NewAndLegacy: indicates that both old and
  new connectivities are allocated, but that access via Item and ItemInternal is
  done using these new connectivities. This mode is therefore close to the
  future operating mode.

In the long term, there will be a 5th value corresponding to the definitive mode
where only the new connectivities are allocated. This will be implemented when
all codes using %Arcane have been validated with the new connectivities.

Depending on how %Arcane is configured, certain values are not possible. If the
configuration is done with **--with-legacy-connectivity**, then the
Arcane::InternalConnectivityPolicy::NewAndLegacy mode is not possible. If
%Arcane is configured with **--without-legacy-connectivity**, then the
Arcane::InternalConnectivityPolicy::Legacy mode is not possible.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_amr_cartesianmesh
</span>
<span class="next_section_button">
\ref arcanedoc_entities_itemtype
</span>
</div>
