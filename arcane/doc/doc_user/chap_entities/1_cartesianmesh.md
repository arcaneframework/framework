# Management of Cartesian Meshes {#arcanedoc_entities_cartesianmesh}

[TOC]

This page describes the management of Cartesian meshes in %Arcane.

\note For now, %Arcane does not automatically handle the recalculation of
structuring information when the mesh changes. You must explicitly call
Arcane::ICartesianMesh::computeDirections() to perform this recalculation.

## Initialization {#arcanedoc_entities_cartesianmesh_init}

To have information about a Cartesian mesh, it is necessary to have an instance
of the Arcane::ICartesianMesh class. To retrieve such an instance, you must use
the Arcane::ICartesianMesh::getReference() method:

```cpp
Arcane::IMesh* mesh = ...;
Arcane::ICartesianMesh* cartesian_mesh = Arcane::ICartesianMesh::getReference(mesh,true);
```

\warning Once the instance is created and before you can use it, it is necessary
to calculate the direction information via the
Arcane::ICartesianMesh::computeDirections() method. This call should only be
made once if the mesh does not change, for example, during code initialization.

```cpp
cartesian_mesh->computeDirections();
```

## Using Directional Information {#arcanedoc_entities_cartesianmesh_direction}

Once this is done, it is possible to get information about the entities for a
given direction. The possible directions are provided by the #eMeshDirection
type. It is also possible to use an integer to specify the direction, where 0
corresponds to the X direction, 1 to the Y direction, and 2 to the Z direction.
For readability reasons, it is recommended to use the enumerated type if
possible.
For example, to retrieve information about cells in the Y direction:

```cpp
using namespace Arcane;
Arcane::CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
Arcane::CellDirectionMng cell_dm(cartesian_mesh->cellDirection(1));
```

\warning The objects managing entities by direction are temporary objects that
should not be retained, particularly from one iteration to the next or when the
mesh changes.

Once a direction is retrieved, it is possible to iterate over all entities in
that direction and, for cells for example, to get the cell before and after:

```cpp
using namespace Arcane;
ENUMERATE_(Cell, icell, cell_dm.allCells()){
  Arcane::Cell cell = *icell;
  Arcane::DirCell dir_cell(cell_dm[icell]); // Directional info for cell
  Arcane::Cell prev_cell = dir_cell.previous(); // Cell before
  Arcane::Cell next_cell = dir_cell.next(); // Cell after.
}
```

For boundary cells, it is possible that \a prev_cell or \a next_cell is null.
This can be tested using the Arcane::Cell::null() method.

Retrieving nodes in a direction is done in the same way.

```cpp
using namespace Arcane;
Arcane::NodeDirectionMng node_dm(cartesian_mesh->nodeDirection(MD_DirX));
ENUMERATE_(Node, inode, node_dm.allNodes()){
  Arcane::Node node = *inode;
  Arcane::DirNode dir_node(node_dm[inode]); // Directional info for node
  Arcane::Node prev_cell = dir_node.previous(); // Node before
  Arcane::Node next_cell = dir_node.next(); // Node after
}
```

For faces, the writing is similar, but instead of retrieving the face before and
after the current face, you can retrieve the cell before and after:

```cpp
using namespace Arcane;
Arcane::FaceDirectionMng face_dm(cartesian_mesh->faceDirection(MD_DirX));
ENUMERATE_(Face, iface, face_dm.allFaces()){
  Arcane::Face face = *iface;
  Arcane::DirFace dir_face(face_dm[iface]);
  Arcane::Cell prev_cell = dir_face.previousCell(); // Cell before the face
  Arcane::Cell next_cell = dir_face.nextCell(); // Cell after the face
}
```

Finally, for cells, it is possible to retrieve directional information about the
nodes of a cell following a direction, via the Arcane::DirCellNode class.

```cpp
using namespace Arcane;
Arcane::CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
ENUMERATE_(Cell, icell, cell_dm.allCells()){
  Arcane::Cell cell = *icell;
  Arcane::DirCellNode cn(cell_dm.cellNode(cell));
  Arcane::Node next_left = cn.nextLeft(); // Node to the left towards the next cell.
  Arcane::Node next_right = cn.nextRight(); // Node to the right towards the next cell.
  Arcane::Node prev_right = cn.previousRight(); // Node to the right towards the previous cell.
  Arcane::Node prev_left = cn.previousLeft(); // Node to the left towards the previous cell.
}
```

Similarly, it is also possible to know the face in front of and behind the cell
in a given direction (this also works in 3D):

```cpp
using namespace Arcane;
Arcane::CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
ENUMERATE_(Cell, icell, cell_dm.allCells()){
  Arcane::Cell cell = *icell;
  Arcane::DirCellFace cf(cell_dm.cellFace(cell));
  Arcane::Face next_left = cf.next(); // Face connected to the next cell.
  Arcane::Face prev_right = cf.previous(); // Face connected to the previous cell.
}
```

To iterate over all directions of a mesh, you can loop as follows:

```cpp
using namespace Arcane;
Integer nb_dir = mesh->dimension();
for( Integer idir=0; idir<nb_dir; ++idir){
  CellDirectionMng cdm(cartesian_mesh->cellDirection(idir));
  ENUMERATE_(Cell, icell, cdm.allCells()){
    ...
  }
}
```

It is possible to know the global number of cells in a given direction via
Arcane::CellDirectionMng::globalNbCell(). Similarly, assuming that the subdomain
decomposition can be represented as a grid, it is possible to know the numbering
in this grid via Arcane::CellDirectionMng::subDomainOffset(). This numbering
starts at 0.

It is also possible to know the number of own cells of the subdomain in a given
direction via Arcane::CellDirectionMng::ownNbCell(). It is also possible to know
the offset in the grid of the first own cell via
Arcane::CellDirectionMng::ownCellOffset().

\warning This information is only accessible if the mesh was generated via the
specific Cartesian generator. In particular, it is not accessible if the mesh
comes from a file. For more information, refer to the description of these
methods.

## Using Cartesian Connectivities {#arcanedoc_entities_cartesianmesh_cartesian_connectivity}

In 2D, it is possible to access the cells around a node and the nodes of the
cell without going through directional connectivities. This is done via the
Arcane::CartesianConnectivity object returned by calling
Arcane::ICartesianMesh::connectivity(). For example:

\snippet CartesianMeshTesterModule.cc SampleNodeToCell

And similarly for cells:

```cpp
using namespace Arcane;
Arcane::CartesianConnectivity cc = cartesian_mesh->connectivity();
ENUMERATE_(Cell, icell, allCells()){
  Arcane::Cell c = *icell;
  Arcane::Node n1 = cc.upperLeft(c); // Node upper left
  Arcane::Node n2 = cc.upperRight(c); // Node upper right
  Arcane::Node n3 = cc.lowerRight(c); // Node lower right
  Arcane::Node n4 = cc.lowerLeft(c); // Node lower left
}
```

These connectivities are also accessible in 3D. The nomenclature is the same as
for 2D connectivities. The prefix topZ is used for the nodes above the same
following the Z direction. For those below, there is no prefix, and therefore
the method name is the same as in 2D. This potentially allows the use of the
same code in 2D and 3D.

```cpp
using namespace Arcane;
Arcane::CartesianConnectivity cc = cartesian_mesh->connectivity();
ENUMERATE_(Cell, icell, allCells()){
  Arcane::Cell c = *icell;
  Arcane::Node n1 = cc.upperLeft(c); // Node below in Z, upper left
  Arcane::Node n2 = cc.upperRight(c); // Node below in Z, upper right
  Arcane::Node n3 = cc.lowerRight(c); // Node below in Z, lower right
  Arcane::Node n4 = cc.lowerLeft(c); // Node below in Z, lower left
  Arcane::Node n5 = cc.topZUpperLeft(c); // Node above in Z, upper left
  Arcane::Node n6 = cc.topZUpperRight(c); // Node above in Z, upper right
  Arcane::Node n7 = cc.topZLowerRight(c); // Node above in Z, lower right
  Arcane::Node n8 = cc.topZLowerLeft(c); // Node above in Z, lower left
}
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities
</span>
<span class="next_section_button">
\ref arcanedoc_entities_snippet_cartesianmesh
</span>
</div>
