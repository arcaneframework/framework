# Geometry Management {#arcanedoc_entities_geometric}

[TOC]

This page describes the classes managing geometry in %Arcane.

The purpose of %Arcane's geometric classes is to provide a unified data
structure for efficiently managing operations on polygons (in 2D) and
polyhedra (in 3D).

## Introduction {#arcanedoc_entities_geometric_introduction}

The supported polygons and polyhedra are defined in the GeomType enumeration.
These are:
- in 2D, triangles, quadrangles, pentagons, and hexagons.
- in 3D, tetrahedrons, pyramids, classic prisms, hexahedrons, pentagonal-based
  prisms (heptahedrons), and hexagonal-based prisms (octahedrons).

%Arcane provides two types of objects for managing geometry.
- the first and simplest type is called a <b>geometric element</b> and contains
  only the coordinates of the nodes of this element. These classes are named
  after the element type followed by \b %Element. For example, for hexahedrons,
  the name is Hexaedron8Element.
- the second type is called a <b>geometric shape</b> and contains, in addition
  to the node coordinates, the coordinates of the face centers, edge midpoints,
  and the center, as well as connectivity information. Geometric shapes are
  managed by the GeomShape class and views on these geometric shapes by the
  GeomShapeView class (formerly GenericElement). Generally, only the view is
  used.

## Geometric Elements {#arcanedoc_entities_geometric_geomelement}

The term <b>geometric element</b> encompasses all classes that manage geometric
elements by retaining only the coordinates of the vertices of these elements.
The different classes are:
- Triangle3Element;
- Quad4Element;
- Pentagon5Element;
- Hexagon6Element;
- Tetraedron4Element;
- Pyramid5Element;
- Pentaedron6Element;
- Hexaedron8Element;
- Heptaedron10Element;
- Octaedron12Element;

They are used in the same way; only the number of coordinates differs:
```cpp
Real3 x0,x1,x2,x3,x4,x5,x6,x7,x8;
Quad4Element quad(x0,x1,x2,x3); // Creation of a quad with initialization
Hexaedron8Element hexa; // Creation of an uninitialized hexa
hexa.init(x0,x1,x2,x3,x4,x5,x6,x7); // Initialization
hexa[5] = Real3(1.2,0.0,0.0); // Changes the value of the 6th vertex
Real3 z = hexa[4]; // Retrieves the value of the 5th vertex
```

Geometric elements are generally used via the concept of a view, similar to
array classes (Array, ArrayView, and ConstArrayView). Therefore, there is a
modifiable view and a constant view for each type of geometric element. To get
the name of the view, simply add \a View or \a ConstView to the class name:
```cpp
Quad4Element quad;
Quad4ElementView quad_view = quad.view();
Quad4ElementConstView quad_const_view = quad.constView();
```

Conversion of an element to a view can be done automatically:

```cpp 
Quad4Element quad;
Quad4ElementView quad_view = quad;
Quad4ElementConstView quad_const_view = quad;
```

## Geometric Shapes {#arcanedoc_entities_geometric_geomshape}

Unlike geometric elements, there is only one class to manage geometric shapes.
This class is called GeomShape and can contain the geometric information of any
mesh type defined in the GeomType enumeration.

A geometric shape contains the coordinates of the nodes, the face centers, the
edge midpoints, and the center of the shape.

The geometric shape is used exclusively via a view on a GeomShape. This view is
called GeomShapeView and contains all the methods to retrieve the necessary
information about the geometric shape. There are also specific views by
geometric type. As with views on geometric elements, these classes are named
after the geometric type suffixed by \a ShapeView:
- Triangle3ShapeView;
- Quad4ShapeView;
- Pentagon5ShapeView;
- Hexagon6ShapeView;
- Tetraedron4ShapeView;
- Pyramid5ShapeView;
- Pentaedron6ShapeView;
- Hexaedron8ShapeView;
- Heptaedron10ShapeView;
- Octaedron12ShapeView;

%Arcane manages two possible uses of geometric shapes:
- the geometric shape associated with a mesh element. For this use, the
  GeomShapeMng class is used, which retains all the necessary information for a
  given mesh (see the documentation for the GeomShapeMng class for its usage and
  initialization). Retrieving a view is done as follows:

```cpp
GeomShapeMng& shape_mng;
Cell cell;
GeomShapeView shape_view;
// Initialization from a mesh \a cell
shape_mng.initShape(shape_view,cell);
```

- an arbitrary geometric shape, which is not directly associated with a mesh
  entity and can be created anywhere. It can be used, for example, to define a
  geometric shape on the control sub-volumes of a mesh. For this case, an
  instance of GeomShape must be used to retain the information. This instance
  must remain valid as long as you wish to use the associated view.
  Initialization is done either with a hexahedron or with a quadrangle (it is
  planned in a later version to be able to initialize with other types). For
  example, for a hexahedron:
```cpp
GeomShape shape;
Hexaedron8Element hexa;
// Initialization from an existing geometric element \a hexa.
Hexaedron8ShapeView shape_view = shape.initFromHexaedron8(hexa);
```

## View Usage {#arcanedoc_entities_geometric_viewusage}

\warning Like all classes that use the concept of a view in %Arcane, views on
geometric objects are only valid as long as the container they originate from
remains valid. In particular, their use must be restricted to passing parameters
between methods, and you must <b>NEVER</b> store a view during a calculation
(such as as a class field, for example).

When you want to use a geometric object, the type of view to use depends on the
coordinates you need:
- if you only need the coordinates of the element's nodes, you must use a view
  on an element. The view must be constant if you are not modifying the element.
  For example, for a method that calculates the volume of a hexahedron, you must
  use an Hexaedron8ElementConstView as a parameter.
- if you need, in addition to the node coordinates, the coordinates of the face
  centers, edge midpoints, or the element's center, you must use a view on a
  shape. If you do not know the exact type of the shape, you must use a
  GeomShapeView. If you know the exact type, you must use the corresponding
  view. For example, for a quadrangle, a Quad4ShapeView.

There is also a GeomShapeOperation class that allows you to obtain a class
implementing IItemOperationByBasicType by providing only the operations for a
given view type (see the GeomShapeOperation documentation for more information)

### Views on Geometric Elements. {#arcanedoc_entities_geometric_geomelementview}

Views on elements should be used wherever possible, particularly instead of
passing \a N coordinates as arguments. For example, for a method calculating the
surface area of a quadrangle, instead of:
```cpp
Real computeSurface2D(const Real3& a0, const Real3& a1,
                      const Real3& a2, const Real3& a3)
{  
  Real3 fst_diag = a2 - a0;
  Real3 snd_diag = a3 - a1;
  return 0.5 * math::crossProduct2D(fst_diag,snd_diag);
}
```

it is better to use:

```cpp
Real computeSurface2D(Quad4ElementConstView quad)
{  
  Real3 fst_diag = quad[2] - quad[0];
  Real3 snd_diag = quad[3] - quad[1];
  return 0.5 * math::crossProduct2D(fst_diag,snd_diag);
}
```

The benefits of using the view are multiple:
- Geometric elements and shapes can easily be converted into a view:
```cpp
// Usage with a geometric element.
Quad4Element my_quad;
computeSurface2D(my_quad);

// Usage with a geometric shape
GeomShapeMng& shape_mng;
Cell cell;
GeomShapeView shape_view;
shape_mng.initShape(shape_view,cell);
computeSurface2D(shape_view.toQuad4Element());
```
- it is possible to create a view from \a N coordinates:
```cpp
// Usage from 4 reals
Real3 a0,a1,a2,a3;
computeSurface2D(Quad4Element(a0,a1,a2,a3));

// Usage from an array of 4 reals
Real3 a[4];
computeSurface2D(Quad4Element(Real3ConstArrayView(4,a)));

// Usage from an entity's coordinates.
VariableNodeReal3& node_coords;
Face face;
computeSurface2D(Quad4Element(node_coords,face));
```
- possibility of specializing operations when using templates, notably
  distinguishing between operations that take the same number of coordinates
  but operate on different elements (for example, between a Quad4 and a
  Tetraedron4)
- eventually, the possibility of easier vectorization.

The view on geometric elements therefore allows unifying several calling
mechanisms and must therefore be used in all cases where possible (i.e.,
always). In particular, methods that take \a N coordinates as arguments can
always be replaced by a method that takes a view of the corresponding element.

### GeomShapeView Usage {#arcanedoc_entities_geometric_geomshapeview}

GeomShapeViews are optimized for geometric calculations within a cell. It is
therefore preferable to use them rather than fetching the node coordinates of a
mesh every time via the IMesh::nodesCoordinates() variable. In particular, they
use a data structure that is optimized for cache management and for
vectorization. Furthermore, they will eventually allow managing geometric shapes
corresponding to finite elements of order 2 or higher.

For example, to retrieve the midpoint of nodes 3 and 4 of a cell:
```cpp
// Classic method
VariableNodeReal3& node_coord = ...;
ENUMERATE_CELL(icell,allCells()){
  Cell cell = *icell;
  Real3 middle = (node_coord[cell.node(3)] + node_coord[cell.node(4)]) / 2.0;
}

// Optimized method.
GeomShapeMng& shape_mng = ...;
GeomShapeView shape;
ENUMERATE_CELL(icell,allCells()){
  shape_mng.initShape(shape,*icell);
  Real3 middle = (shape.node(3) + shape.node(4)) / 2.0;
}
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_itemtype
</span>
<span class="next_section_button">
\ref arcanedoc_entities_tools
</span>
</div>
