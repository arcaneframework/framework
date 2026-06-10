# Modifications in entity handling {#arcanedoc_item_handling_news}

This page groups the modifications related to managing entity connectivities
across different versions of %Arcane.

[TOC]

## Modifications in version 3.7 {#arcanedoc_news_37}

Version 3.7 of %Arcane introduces several modifications to how mesh entities are
managed. These modifications aim to:

- reduce memory footprint
- allow access to certain information about entities on accelerators, such as
  Arcane::Item::owner().
- improve performance

To meet these objectives, the following modifications were made:

- The Arcane::ItemSharedInfo class no longer contains specific information about
  an entity. Therefore, only a single instance of this class is needed per
  Arcane::IItemFamily. The method
  Arcane::mesh::ItemFamily::commonItemSharedInfo() allows this instance to be
  retrieved.
- Since there is now only one instance of Arcane::ItemSharedInfo, using
  Arcane::ItemInternal becomes optional when creating an instance of
  Arcane::Item or one of its derived classes. It is now possible to create an
  instance of these classes using only an Arcane::ItemLocalId and an
  Arcane::ItemSharedInfo*. The internal class Arcane::ItemBaseBuildInfo is used
  for this purpose.
- An Arcane::ItemBase class was created. It contains only an Arcane::ItemLocalId
  and an Arcane::ItemSharedInfo*. It serves as the base class for Arcane::Item
  and Arcane::ItemInternal. This class is internal to Arcane but does not
  allow modification of entity values (for example, it contains the method
  Arcane::ItemBase::owner() but not the method `setOwner()`). Thanks to this
  class, Arcane::Item no longer depends on Arcane::ItemInternal and can directly
  retrieve information via Arcane::ItemSharedInfo, which avoids a level of
  indirection when accessing connectivities, for example.
- Iterators over entities (Arcane::ItemEnumerator) have been modified so that
  they no longer use Arcane::ItemInternal during each incrementation, which can
  slightly improve performance by avoiding an additional indirection every time.
- The data previously carried by Arcane::ItemInternal (such as `owner()`,
  `flags()`) is now stored in array variables (Arcane::VariableArrayInt32 or
  Arcane::VariableArrayInt16) managed by the entity family.

With these modifications, it will eventually be possible to completely eliminate
the use of Arcane::ItemInternal.

However, this class is often used in code, so this change must be gradual.
Notably, the method Arcane::IItemFamily::itemsInternal() is used to retrieve an 
instance of Arcane::Item from an Arcane::ItemInternalArrayView (which is the 
type returned by this method).

To prepare for this and keep the code compatible, a new class
Arcane::ItemInfoListView (and entity-specific derived classes like
Arcane::CellInfoListView, Arcane::DoFInfoListView) allows retrieving information
about entities for which Arcane::IItemFamily::itemsInternal() was previously
used.

It is possible to modify the current code as follows:

~~~cpp
Arcane::IItemFamily* cell_family = ...;
Arcane::ItemInternalArrayView cells = cell_family->itemsInternal();
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

This code should be replaced by this:

~~~cpp
Arcane::IItemFamily* cell_family = ...;
Arcane::CellInfoListView cells(cell_family);
Arcane::Int32 my_local_id = ...;
Arcane::Cell my_cell = cells[my_local_id];
~~~

Later, if the unique instance of Arcane::ItemSharedInfo per family is created in
unified memory, it will be possible to access entity information on the
accelerator.

## Modifications in version 3.10

%Arcane frequently uses lists of Arcane::ItemLocalId to manage mesh entities,
which can be converted into lists of Arcane::Int32. This is used in the
following cases, for example:

- List of entities in a group (Arcane::ItemGroup) or an entity vector
  (Arcane::ItemVector)
- List of entities connected to another entity (for example,
  Arcane::Cell::nodes())

For these two cases, the internal structure is managed in the same way (via an
instance of Arcane::ItemVectorView), and %Arcane internally maintains objects of
type Arcane::Int32ConstArrayView that can be accessed directly by the developer
(for example, via Arcane::ItemVectorView::localIds()).

To more efficiently manage connectivities, especially in the Cartesian case, and
reduce memory footprint, it is necessary to evolve how these localId() lists are
stored. To enable these evolutions, two things must be modified in %Arcane:

1. Separate the management of entity connectivity from that of group entity
   lists.
2. Hide the internal structure used to store these localId() lists.

This involves changing certain mechanisms for accessing this information, which
are detailed below.

### Separating entity connectivity management from group entity list management

This means that the methods for accessing connectivities and those for accessing
group entities do not return the same type of object. In version 3.9 of %Arcane,
the connectivity access methods were therefore modified and now return an
instance of Arcane::ItemConnectedListViewT instead of an instance of
Arcane::ItemVectorViewT.

There is currently a conversion operator between Arcane::ItemConnectedListViewT
and Arcane::ItemVectorViewT to make the code compatible with existing usage.

This also impacts macros such as ENUMERATE_(), ENUMERATE_CELL() or
ENUMERATE_NODE(), which are now reserved for iterations over Arcane::ItemGroup
or Arcane::ItemVector. Currently, there are several ways to iterate over
entities in another connectivity. For example:

~~~cpp
Arcane::CellGroup cell_group = ...;
ENUMERATE_(Cell,icell,cell_group){
  Arcane::Cell cell = *icell;
  // (1) Iteration with ItemEnumerator
  for( Arcane::NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode ){
    Arcane::Node node = *inode;
    info() << "Node uid=" << node.uniqueId();
  }
  // (2) Iteration with ENUMERATE_
  ENUMERATE_(Node,inode,cell.nodes()){
    Arcane::Node node = *inode;
    info() << "Node uid=" << node.uniqueId();
  }
  // (3) Iteration with 'for-range'
  for( Arcane::Node node : cell.nodes()){
    info() << "Node uid=" << node.uniqueId();
  }
}
~~~

Mechanism (3) should be preferred. Eventually, mechanism (1) will disappear
because the Arcane::ItemEnumerator type will be reserved for iterations over
groups. Mechanism (2) might continue to be available but will be less performant
than mechanism (3).

To also avoid any risk of future incompatibility, it is preferable not to
directly use the returned iterator types but to use the `auto` keyword instead.

### Hiding the internal structure managing localId() lists

To hide these structures, %Arcane classes that manage entity lists will no
longer return types such as Arcane::Int32ConstArrayView. For example, methods
such as Arcane::ItemVectorView::localIds() or
Arcane::ItemIndexArrayView::localIds() will disappear. To be compatible with
existing usage, the methods Arcane::ItemVectorView::fillLocalIds() and
Arcane::ItemIndexArrayView::fillLocalIds() have been added to allow filling
an array with the list of localIds().
