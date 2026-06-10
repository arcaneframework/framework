# Basic Structures and Types {#arcanedoc_getting_started_basicstruct}

[TOC]

## Basic Types {#arcanedoc_getting_started_basicstruct_types}

%Arcane provides a set of basic types, corresponding either to an existing C++
type (such as *int*, *double*) or to a class (such as \arcane{Real2}). These
types are used for all common operations as well as for variables. For example,
when you want to declare an integer, you must use \arccore{Integer} instead of
*int* or *long*. This allows you to change the size of these types (for example,
using 8-byte integers instead of 4) without modifying the source code.

The basic types are:

<table>
<tr><td><b>Class Name</b></td><td><b>Specification Mapping</b></td></tr>
<tr><td>\arccore{Integer}   </td><td> 32-bit signed integer </td></tr>
<tr><td>\arccore{Int16}     </td><td> 16-bit signed integer </td></tr>
<tr><td>\arccore{Int32}     </td><td> 32-bit signed integer </td></tr>
<tr><td>\arccore{Int64}     </td><td> 64-bit signed integer </td></tr>
<tr><td>\arcane{Byte}       </td><td> represents an 8-bit character </td></tr>
<tr><td>\arccore{Real}      </td><td> IEEE 754 real </td></tr>
<tr><td>\arcane{Real2}      </td><td> 2D coordinate, vector of two reals </td></tr>
<tr><td>\arcane{Real3}      </td><td> 3D coordinate, vector of three reals </td></tr>
<tr><td>\arcane{Real2x2}    </td><td> 2D tensor, vector of four reals </td></tr>
<tr><td>\arcane{Real3x3}    </td><td> 3D tensor, vector of nine reals </td></tr>
<tr><td>\arccore{String}    </td><td> UTF-8 formatted character string </td></tr>
</table>

The floats (\arccore{Real}, \arcane{Real2}, \arcane{Real2x2}, \arcane{Real3},
\arcane{Real3x3}) use double-precision IEEE 754 reals and are stored in 8 bytes.

## Mesh Entities {#arcanedoc_getting_started_basicstruct_meshitem}
There are 4 types of basic entities in a mesh: nodes, edges, faces, and cells.
Each of these types corresponds to a C++ class in %Arcane. For each entity type,
there is a *group* type that manages a set of entities of that type. The class
that manages a group of an entity is named after the entity with the suffix
*Group*. For example, for nodes, this is \arcane{NodeGroup}.

<table>
<tr><td><b>Class Name</b></td><td><b>Specification Mapping</b></td></tr>
<tr><td>\arcane{Node}      </td><td> a node </td></tr>
<tr><td>\arcane{Cell}      </td><td> a cell </td></tr>
<tr><td>\arcane{Face}      </td><td> a 3D face, a 2D edge</td></tr>
<tr><td>\arcane{Edge}      </td><td> a 3D edge</td></tr>
<tr><td>\arcane{Particle}  </td><td> a particle</td></tr>
<tr><td>\arcane{DoF}       </td><td> a degree of freedom</td></tr>
<tr><td>\arcane{NodeGroup} </td><td> a group of nodes </td></tr>
<tr><td>\arcane{CellGroup} </td><td> a group of cells </td></tr>
<tr><td>\arcane{FaceGroup} </td><td> a group of faces </td></tr>
<tr><td>\arcane{EdgeGroup} </td><td> a group of edges </td></tr>
<tr><td>\arcane{ParticleGroup} </td><td> a group of particles</td></tr>
<tr><td>\arcane{DoFGroup} </td><td> a group of degrees of freedom</td></tr>
</table>

\note
The faces \arcane{Face} are the N-1 dimensional entities where N is the
dimension of the cells. In dimension 2, faces therefore correspond to edges, and
in dimension 3 to the faces of polyhedra. This allows numerical algorithms to
traverse the mesh regardless of its dimension. The edge entity (\arcane{Edge})
only exists for 3D meshes and then corresponds to an edge.

Each mesh entity corresponds to an instance of a class. For example, if the mesh
contains 15 cells, there are 15 instances of the \arcane{Cell} type. Each class
provides a certain number of operations allowing instances to be linked
together. For example, the \arcane{Cell::node}(\arccore{Int32}) method of the
\arcane{Cell} class allows you to retrieve the i-th node of this cell.
Similarly, the \arcane{Cell::nbNode()} method allows you to retrieve the number
of nodes in the cell. For more information on the supported operations, you need
to refer to the online documentation of the corresponding classes
(\arcane{Node}, \arcane{Edge}, \arcane{Face}, \arcane{Cell}, \arcane{Particle},
\arcane{DoF}).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started_about
</span>
<span class="next_section_button">
\ref arcanedoc_getting_started_iteration
</span>
</div>
