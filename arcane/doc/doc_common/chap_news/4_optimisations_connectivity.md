# Entity Connectivity Optimizations {#arcanedoc_new_optimisations_connectivity}

[TOC]

## Context

In the current version of %Arcane (3.x), the connectivity of entities and groups
of entities is managed as if everything were unstructured, even if the mesh is
Cartesian or structured. Specifically, all connectivity information between
entities is retained, which consumes memory. This memory cost is difficult to
reduce when the mesh is truly unstructured, but it could be reduced for
structured and Cartesian meshes. Particularly in these latter cases, we often
have many elements and few variables, and proportionally, the memory cost of
retaining connectivity is significant.

It was therefore decided to modify the internal management of this connectivity
information in %Arcane to reduce memory consumption and allow for further
optimizations.

These optimizations must meet the following constraints:

- Impact the codes using %Arcane as little as possible; therefore, these
  evolutions must be progressive to allow the codes to make the necessary
  changes (as was the case with the new connectivities for version 3.0).
- The connectivity management mechanism must remain generic to apply to all mesh
  types with the current looping and iteration mechanisms.

## Proposed Optimizations

The proposed optimizations are based on the assumption that the connectivity and
group schema is often the same for a large number of entities. The proposed
optimizations are divided into two groups:

- optimizations on connectivities (e.g., \arcane{Cell::nodes()})
- optimizations on entity groups (\arcane{ItemGroup}).

### Connectivity Optimizations

Currently, connectivities are managed by the class
\arcane{mesh::IncrementalItemConnectivity}, and there are three arrays (of
\arcane{Int32}) to store the connectivity of one entity to another:

1. an array containing the connectivity. The size of this array is at least
   equal to the sum of the connected entities. For example, if we have 40
   hexahedrons, its size is 8x40 elements.
2. an array per entity indicating how many connected entities it has.
3. an array per entity indicating the position in array (1) of the first
   connected element.

The proposed optimization is based on the assumption that for a given entity,
the `localId()` of the entities connected to it are often the same relative to
the `localId()` of the first connected entity. Instead of retaining all
connectivity, we can therefore only retain the connectivity schema as well as
the `localId()` of the first entity.

For this optimization, we must therefore:

1. retain the `localId()` of the first connected entity for each entity
2. when accessing the i-th connected entity, add the `localId()` of the first
   connected entity to the stored value.

Operation (2) will have a negligible computational time cost, as the value to be
added will be retained during iteration in the same way as the number of
connected entities. So, for (1), we only have the addition of an 'Int32' for
each entity, but this will be compensated by the reuse of the connectivity.

For example, for a Cartesian mesh of 2 rows and 4 columns, we currently have the
following numbering:

```
10---11---12---13---14
| 4  | 5  | 6  | 7  |
5----6----7----8----9
| 0  | 1  | 2  | 3  |
0----1----2----3----4
```

If I take the cell/node connectivity, the three arrays contain the following
values:

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
2. 4         4         4         4         4           4           4           4
3. 0         4         8        12         16          20         24          28
```

If I apply the optimization, I add the following array containing the
`localId()` of the first entity:

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
2. 4         4         4         4         4           4           4           4
3. 0         4         8        12         16          20         24          28
4. 0         1         2         3         5           6           7           8
```

I subtract the value associated with (4) from (1)

```
1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
4. 0         1         2         3         5           6           7           8

1. 0 5 6 1 | 1 6 7 2 | 2 7 8 3 | 3 8 9 4 | 5 10 11 6 | 6 11 12 7 | 7 12 13 8 | 8 13 14 9
-  0 0 0 0   1 1 1 1   2 2 2 2   3 3 3 3   5 5  5  5   6  6  6 6   7 7  7  7 | 8  8  8 8
=  0 5 6 1   0 5 6 1   0 5 6 1   0 5 6 1   0 5  6  1   0  5  6 1   0 5  6  1   0  5  6 1
```

We can see that for this case (ideal), the schema is the same. We can therefore
only keep it once, and we will have the following values for the connectivity:

```
1. 0 5 6 1
2. 4         4         4         4         4           4           4           4
3. 0         0         0         0         0           0           0           0
4. 0         1         2         3         5           6           7           8
```

If `N` is the number of elements, we go from a memory consumption of
`(N*4 + N + N)` to `(4 + N + N +N)`, which is from `6*N` to `3*N`. In the 3D
case, we go from `(N*8 + N + N)` to `(8 + N + N + N)`, which is from `10*N` to
`3*N`.

\note We could also consider retaining the number of connected entities to an
entity in (1), which would allow us to delete array (2).

In Cartesian meshes, the schema for cells and nodes is independent of the number
of cells and nodes. However, for faces, there is a schema per row and one per
column. So, for example, for a 100x30x20 mesh, there are 32x20 schemas for the
faces, resulting in 640 values instead of 60000 without the optimization.

In addition to reducing memory consumption, this will allow for better cache
utilization.

In the worst case, if there is no recurrent schema, the connectivity will
consume an additional `N` \arcane{Int32}. This is likely only the case for
meshes composed of arbitrary triangles (2D) or tetrahedrons (3D), which is not
the case for CEA and IFPEN applications.

These mechanisms can also be applied to classes specifically managing Cartesian
data, which also have similar access schemas.

### Approach to Implement These Optimizations

In order to proceed with these optimizations transparently, we need an iterator
over the connectivities that is different from the iterator over the entities,
which is not the case currently, as both use \arcane{ItemVectorView} and
\arcane{ItemEnumerator} as container and iterator.
This allows us to code as follows:

```{cpp}
Arcane::CellGroup cells;
ENUMERATE_(Cell,icell,cells){
  Arcane::Cell cell = *icell;
  ENUMERATE_(Face,iface,cell.faces()){
  }
}
```

This must therefore be prohibited for connectivities. We can add a specific
macro to enumerate over connectivities or replace it with the `for-loop`:

```{cpp}
Arcane::CellGroup cells;
ENUMERATE_(Cell,icell,cells){
  Arcane::Cell cell = *icell;
  // for-loop
  for ( Arcane::Face face : cell.faces()){
  }
}
```

### Group Optimizations

The same principle as for connectivities can be applied to groups.
Currently, we retain a simple indirection list. For a group with M elements, we
must retain M \arcane{Int32}.

It would be possible to decompose the list of entities into blocks and retain 3
values for each block corresponding to arrays (2), (3), and (4) of the
connectivities. Optionally, the number of elements in each block can also be
pooled in the indirection list.

With a block size of 128, for example, we must retain `(3*M/128)` values for the
indirection information. But in the case where the array values are contiguous,
we just need to retain an additional 128 values. This again allows for memory
savings and better cache utilization. This mechanism also has the advantage of
being easily usable on accelerators.

### Approach to Implement These Optimizations

However, this requires two modifications in %Arcane:
- making the ability to retrieve the `localIds()` of groups obsolete
- transforming the `ENUMERATE_` macro to perform two loops: a loop over the
  blocks followed by a loop for each block. This therefore requires changing the
  \arcane{ItemEnumerator} class to manage blocks or creating another one. One
  possibility is, for example, to have two new classes `ItemBlockVectorView` and
  `ItemBlockEnumerator`. The current case using \arcane{ItemVectorView} and
  \arcane{ItemEnumerator} would be a specific case of a block where the number
  of values corresponds to the number of elements in the vector and the
  `localId()` offset would be zero.

It must be possible to still use a double loop in `ENUMERATE_`, even if the
second loop is fixed at 1 at compile time in the case of
\arcane{ItemVectorView}, for example.

### Planning

The modifications to change these connectivities will begin in version 3.10 of
%Arcane (June 2023).

The first optimization planned will concern entity connectivities. To this end,
the following modifications are made:

1. the methods allowing access to the `localId()` arrays of the connectivities
   will be made obsolete. This concerns the classes \arcane{ItemEnumerator},
   \arcane{ItemEnumeratorBase}, \arcane{ItemConnectedListView}.
2. The use of the \arcane{ItemInternal} class becomes obsolete.
   \arcane{impl::ItemBase} must be used instead. All methods that also took
   \arcane{ItemInternalArrayView}, \arcane{ItemInternalPtr}, or
   \arcane{ItemInternalList} also become obsolete.

In general, user codes of %Arcane are minimally impacted by point (1) because
the concerned structures are rather internal to Arcane. Since it is necessary to
remove these methods to implement compressed connectivities, it is planned to
permanently remove these methods in December 2023.

For point (2), which potentially concerns more parts of the code, it is planned
to remove the concerned methods in June 2024.

The second phase of optimization concerning entity groups will follow.
