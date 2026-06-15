# Variable {#arcanedoc_core_types_axl_variable}

[TOC]

A variable is a value manipulated by the code and managed by %Arcane. For
example, volume and speed are variables. They are characterized by a **name**, a
**data type**, a **support**, and a **dimension**.

Variables are declared inside a module/service, within the module/service
descriptor (for the rest of this page, we will say that module = service, to
simplify the writing).

If two modules use variables with the same name, their values will be implicitly
shared. This is how modules communicate their information.

## Characteristics

### Data Type {#arcanedoc_core_types_axl_variable_types}

The **data type** of variables must be chosen from the following:

| C++ Name          | Type                                 |
|-------------------|--------------------------------------|
| \arccore{Integer} | 32-bit signed integer                |
| \arccore{Int16}   | 16-bit signed integer                |
| \arccore{Int32}   | 32-bit signed integer                |
| \arccore{Int64}   | 64-bit signed integer                |
| \arcane{Byte}     | represents an 8-bit character        |
| \arccore{Real}    | IEEE 754 real                        |
| \arcane{Real2}    | 2D coordinate, vector of two reals   |
| \arcane{Real3}    | 3D coordinate, vector of three reals |
| \arcane{Real2x2}  | 2D tensor, vector of four reals      |
| \arcane{Real3x3}  | 3D tensor, vector of nine reals      |
| \arccore{String}  | unicode character string             |

The floats (*Real*, *Real2*, *Real2x2*, *Real3*, *Real3x3*) are double-precision
reals (stored in 8 bytes).

### Support {#arcanedoc_core_types_axl_variable_support}

The **support** corresponds to the entity that holds the variable, on which the
variable is defined. These variables that apply to mesh elements are called
**quantities**.

| C++ Name          | Support                                     |
|-------------------|---------------------------------------------|
| (empty)           | variable defined globally (e.g.: time step) |
| \arcane{Node}     | mesh node                                   |
| \arcane{Face}     | mesh face                                   |
| \arcane{Cell}     | mesh cell                                   |
| \arcane{Particle} | mesh particle                               |
| \arcane{DoF}      | degree of freedom                           |

### Dimension {#arcanedoc_core_types_axl_variable_dim}

The **dimension** can be:

| C++ Name   | Dimension |
|------------|-----------|
| **Scalar** | scalar    |
| **Array**  | 1D array  |
| **Array2** | 2D array  |

Since version 3.8.11 of %Arcane, it is also possible to declare
multi-dimensional variables on the mesh.

### C++ Class for Scalar, 1D, or 2D Variables {#arcanedoc_core_types_axl_variable_cppclass}

For scalar, 1D, or 2D variables, it is easy to obtain the C++ class
corresponding to a given data type, support, and dimension. The class name is
constructed as follows:

**Variable** + \ref arcanedoc_core_types_axl_variable_support +
\ref arcanedoc_core_types_axl_variable_dim +
\ref arcanedoc_core_types_axl_variable_types

For example, for a variable representing an array of integers,
**VariableArrayInteger**, or for a variable representing a real,
**VariableScalarReal**.

When a scalar variable is defined on a mesh entity, the support (*Scalar*) is
not specified. For example, for a variable representing a real on cells,
**VariableCellReal**.

All combinations are possible with the following exceptions:
- *character string* type variables, which only exist for scalar and array types
  but not on mesh elements (for performance reasons).
- Dimension 2 variables cannot have a support (it is not possible to have 2D
  variables on mesh elements, for example).

The following table gives some examples of variables:

| Name C++                          | Description                      |
|-----------------------------------|----------------------------------|
| \arcane{VariableScalarReal}       | a real                           |
| \arcane{VariableScalarInteger}    | an integer                       |
| \arcane{VariableArrayInteger}     | Array of integers                |
| \arcane{VariableArrayReal3}       | Array of 3D coordinates          |
| \arcane{VariableNodeReal2}        | 2D coordinate at nodes           |
| \arcane{VariableFaceReal}         | Real at faces                    |
| \arcane{VariableFaceReal3}        | 3D coordinate at faces           |
| \arcane{VariableFaceArrayInteger} | Array of integers at faces       |
| \arcane{VariableCellArrayReal}    | Array of reals at cells          |
| \arcane{VariableCellArrayReal3}   | Array of 3D coordinates at cells |
| \arcane{VariableCellArrayReal2x2} | Array of 2D tensors at cells     |

For multi-dimensional variables, there is no `typedef` because the number of
possibilities is infinite.

## Declaration {#arcanedoc_core_types_axl_variable_declare}

Variable declaration is done through the module or service descriptor (AXL file)
or directly in the code.

### Declaration in AXL files

\note Variables declared in the AXL file are always associated with the default
mesh of the module or service. It is not possible to declare subdomain variables
in these files (but it is always possible to declare them explicitly in the
module or service class).

In the AXL file, variable declaration is done in the \c variables tag. The
following example shows the declaration in the \c Test module of a real variable
on cells called \c Pressure and a real variable on nodes called \c NodePressure.

```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <description>Descripteur du module Test</description>
  <variables>
    <variable field-name="pressure" name="Pressure" data-type="real"
              item-kind="cell" dim="0" dump="true" need-sync="true"/>
    <variable field-name="node_pressure" name="NodePressure" data-type="real"
              item-kind="node" dim="0" dump="true" need-sync="true"/>
  </variables>

  <entry-points>
  </entry-points>

  <options>
  </options>
</module>
```

#### Variable Name, Data Type, and Support

The following attributes are mandatory and allow defining the variable's name,
data type, and support:

<table>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
<tr>
<td> **name** </td>
<td>name of the variable. By convention, it starts with a capital letter. Valid
characters are alphabetic characters **[a-zA-Z]**, digits (except at the
beginning of the name), and the underscore character. Variables whose name
starts with *Global* and *%Arcane* are reserved for %Arcane.
</td>
</tr>
<tr>
<td> **field-name** </td>
<td>computer name of the variable field in the generated class. This name must
be a valid C++ name. By convention, if the variable is called *NodePressure*,
for example, the field will be **node_pressure**. In the generated class, the
specified name will be prefixed with **m_** (the \c m_ prefix corresponds to
the "coding rules" in %Arcane for class attributes).
</td>
</tr>
<tr>
<td> **item-kind** </td>
<td>support of the variable. Possible values are \c node, \c edge, \c face,
\c cell, \c particle, \c dof, or \c none. The value \c none indicates that it
is not a mesh variable. If the type is \c particle or \c dof, the
**family-name** attribute must be specified.
</td>
</tr>

<tr>
<td> **data-type** </td>
<td>data type of the variable. It can be chosen from *integer*, *int16*,
*int32*, *int64*, *real*, *string*, *real2*, *real3*, *real2x2*, *real3x3*.
The *string8* type is not usable for mesh variables (those with a support).
</td>
</tr>

<tr>
<td> **family-name** </td>
<td>name of the entity family (\arcane{IItemFamily} to which the variable is
associated. This attribute is only valid if **item-kind** is \c particle or
\c dof.
</td>
</tr>

</table>

#### Variable Dimensions

The following attributes allow defining the variable's dimension.

\note The values **shape-dim**, **extent0**, and **extent1** are only available
starting from version 3.8.11 of %Arcane.

If none of the following attributes are defined, the variable is considered a
scalar variable as if the value **dim="0"** had been specified.

You must specify either **dim** (for classical scalar, 1D, or 2D variables) or
**shape-dim** (for multi-dimensional variables).

<table>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>

<tr>
<td> **dim** </td>
<td>The dimension of the variable for scalar, 1D, or 2D variables. Possible
values are *0* for a scalar variable, *1* for a one-dimensional array variable,
and *2* for a two-dimensional array. Two-dimensional arrays are not supported
for mesh variables.
</td>
</tr>

<tr>
<td> **shape-dim** </td>
<td>The dimension of the variable for multi-dimensional variables. Possible
values are 0, 1, 2, or 3.
</td>
</tr>

<tr>
<td> **extent0** </td>
<td>Number of elements for a multi-dimensional variable whose elements are
vectors, or number of rows for a multi-dimensional variable whose elements are
matrices. This attribute can only be used if **shape-dim** is present.
</td>
</tr>

<tr>
<td> **extent1** </td>
<td>Number of columns for a multi-dimensional variable whose elements are
matrices. This attribute can only be used if **shape-dim** is present.
</td>
</tr>

</table>

#### Variable Properties

The following attributes are optional and allow specifying properties on the
variable:

<table>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>

<tr>
<td> **dump** </td>
<td>allows choosing whether a variable's values are saved when the code stops.
In this case, the saved values will, of course, be read back when execution
resumes. Some recalculated variables do not need to be saved; in this case, the
*dump* attribute is false. This is the case when a variable's value is not
useful from one iteration to the next.
</td>
</tr>

<tr>
<td> **need-sync** </td>
<td>indicates whether the variable must be synchronized between sub-domains. It
is just an indication that can be used during checks.
</td>
</tr>

</table>

When compiling the module descriptor using %Arcane (with \c axl2cc - cf
previously),
the variables are registered in the architecture database.

#### Example of variable declaration in AXL files

```xml

<!-- Scalar variable on 'Real' nodes -->
<variable field-name="node_mass" name="NodeMass"
          data-type="real" item-kind="node" dim="0"/>
  
<!-- Scalar variable on 'Int64' cells -->
<variable field-name="cell_unique_id" name="UniqueId"
          data-type="int64" item-kind="cell" dim="0"/>

<!-- 1D variable on 'Real3' cells -->
<variable field-name="cell_cqs" name="CellCQS"
          data-type="real3" item-kind="cell" dim="1"/>

<!-- Scalar variable on 'Real' particles in the 'ArcaneParticles' family -->
<variable field-name="particle_temperature" name="Temperature"
          family-name="ArcaneParticles"
          data-type="real" item-kind="particle" dim="0"/>

<!-- 0D multi-dim variable on 'Real' cells -->
<variable field-name="mdvar0d" name="MDVar0D"
          data-type="real" item-kind="cell" shape-dim="0"/>

<!-- 1D multi-dim variable on 'Real' cells -->
<variable field-name="mdvar1d" name="MDVar1D"
          data-type="real" item-kind="cell" shape-dim="1"/>

<!-- 2D multi-dim variable on 'Real' cells -->
<variable field-name="mdvar2d" name="MDVar2D"
          data-type="real" item-kind="cell" shape-dim="2"/>

<!-- 2D multi-dim variable on 'Real' cells viewed as a 3D variable -->
<variable field-name="mdvar2d_as_3d" name="MDVar2D"
          data-type="real" item-kind="cell" shape-dim="3"/>

<!-- 0D multi-dim variable on NumVector<Real,2> cells -->
<variable field-name="mdvar0d_vector2" name="MDVar0DVector2"
          data-type="real2" item-kind="cell" shape-dim="0"/>

<!-- 1D multi-dim variable on NumVector<Real,3> faces -->
<variable field-name="mdvar1d_vector3" name="MDVar1DVector3"
          data-type="real3" item-kind="face" shape-dim="1"/>

<!-- 2D multi-dim variable on NumVector<Real,4> faces -->
<variable field-name="mdvar2d_vector4" name="MDVar2DVector4"
          data-type="real" item-kind="face" shape-dim="2" extent0="4"/>

<!-- 0D multi-dim variable on NumMatrix<Real,2,2> nodes -->
<variable field-name="mdvar0d_matrix2x2" name="MDVar0DMatrixReal2x2"
          data-type="real2x2" item-kind="node" shape-dim="0"/>

<!-- 1D multi-dim variable on NumMatrix<Real,3,3> cells -->
<variable field-name="mdvar1d_matrix3x3" name="MDVar1DMatrixReal3x3"
          data-type="real3x3" item-kind="cell" shape-dim="1"/>

<!-- 1D multi-dim variable on NumMatrix<Real,2,6> cells -->
<variable field-name="mdvar1d_matrix2x6" name="MDVar1DMatrix2x6"
          data-type="real" item-kind="cell" shape-dim="1" extent0="2"
          extent1="6"/>

<!-- 0D multi-dim variable on NumMatrix<Real,3,2> cells -->
<variable field-name="mdvar0d_matrix3x2" name="MDVar0DMatrix3x2"
          data-type="real" item-kind="cell" shape-dim="0" extent0="3"
          extent1="2"/>
```

## Variable Usage {#arcanedoc_core_types_axl_variable_use}

### Usage of Scalar, 1D, and 2D Variables

The way a variable is used is identical regardless of its type and depends only
on its kind.

#### Scalar Variables

Scalar variables are used via the template class \arcane{VariableRefScalarT}.

There are only two ways to use them:

- "read the value": this is done using the operator()
  (\arcane{VariableRefScalarT::operator()()}). This operator returns a constant
  reference to the value stored in the variable. It can be used wherever a
  variable's type value is needed.
- "change the value": this is done using the operator=
  (\arcane{VariableRefScalarT::operator=()})

For example, with the variable \c m_time of type \c VariableScalarReal:

```cpp
m_time = 5.0;        // affecte la valeur 5. à la variable m_time
double z = m_time(); // récupère la valeur de la variable et l'affecte à z.
cout << m_time();    // imprime la valeur de m_time
```

It is important not to forget the parentheses when accessing the variable's
value.

#### Array Variables

Array variables are used via the template class \arcane{VariableRefArrayT}.

Their operation is quite similar to the STL \c vector class. The array is sized
using the \arcane{VariableRefArrayT::resize()} method, and each element of the
array can be accessed by the \arcane{VariableRefArrayT::operator[]()} operator,
which returns a reference to the element type.

For example, with the variable *m_times* of type \arcane{VariableArrayReal}:

```cpp
Arcane::VariableArrayReal m_times = ...;
m_times.resize(5);         // resizes the array to contain 5 elements
m_times[3] = 2.0;          // assigns the value 2.0 to the 4th element of the array
cout << m_times[0];        // prints the value of the first element
```

#### Mesh Scalar Variables

These are variables defined on mesh elements (nodes, faces, or cells) with a
value per element. These variables are defined by the template class
\arcane{MeshVariableScalarRefT}.

Their operation is quite similar to that of a standard C array. The
\arcane{VariableRefArrayT::operator[]()} operator is used to retrieve a
reference to the variable type for a given mesh element. This operator is
overloaded to accept an iterator over a mesh element.

Quantities are declared and used similarly regardless of the mesh element type.
They are automatically sized during initialization to the number of mesh
elements of the variable's kind.

For example, with the variable *m_volume* of type \arcane{VariableCellReal}:

```cpp
Arcane::VariableCellReal m_volume = ...;
ENUMERATE_(Cell, icell, allCells()) {
  m_temperature[icell] = 2.0;     // Assigns the value 2.0 to the volume of the current cell
  cout << m_temperature[icell];   // Prints the volume of the current cell

  // it is possible to perform the same operations with the cell
  // ATTENTION this is less performant
  m_temperature[icell] = 2.0;  // Assigns the value 2.0 to the volume of the 'cell'
  cout << m_temperature[icell];    // Prints the volume of the 'cell'
}
```

#### Mesh Array Variables

These are variables defined on mesh elements (nodes, faces, or cells) with an
array of values per element. These variables are defined by the template class
\arcane{MeshVariableArrayRefT}.

The operation of these variables is identical to that of mesh scalar variables,
but the \arcane{MeshVariableArrayRefT::operator[]()} operator returns an array
of values of the variable type.

It is possible to change the number of elements in the second dimension of this
array using the \arcane{MeshVariableArrayRefT::resize()} method.

For example, with the variable *m_temperature* of type
\arcane{VariableCellArrayReal}:

```cpp
Arcane::VariableCellArrayReal m_temperature = ...;
m_temperature.resize(3); // Each cell will have 3 temperature values
ENUMERATE_(Cell, icell, allCells()) {
  // Assigns the value 2.0 to the first temperature of the current cell
  m_temperature[icell][0] = 2.0;

  // Prints the 2nd temperature of the current cell
  info() << m_temperature[icell][1];

  // it is possible to perform the same operations with the cell

  // Declares a reference to a cell.
  Cell cell = *icell;
  // Assigns the value 2.0 to the first temperature of the 'cell'
  m_temperature[cell][0] = 2.0;

  // Assigns the value 4.0 to the 3rd temperature of the 'cell'
  m_temperature(cell,2) = 4.0;

  // Prints the 2nd temperature of the 'cell'
  info() << m_temperature[cell][1];
}
```

### Usage of Multi-Dimensional Variables on the Mesh {#arcanedoc_core_types_axl_md_variable_use}

\warning Multi-dimensional variables are an experimental feature and the API is
not yet finalized. The current code is not guaranteed to be usable in a later
version of %Arcane. Similarly, certain mechanisms involving these variables
(such as dumping) may not be fully operational.

Starting from version 3.8.11 of %Arcane, it is possible to use multi-dimensional
variables on the mesh. These variables unify the use of variables regardless of
their dimension and allow support for up to 3 dimensions (note that this value
may probably be increased in a future version).

\note For now, these variables must have the \arcane{Real} data type.

There are three types of elements for multi-dimensional variables:
- scalars \arcane{MeshMDVariableRefT}
- vectors \arcane{MeshVectorMDVariableRefT}
- matrices \arcane{MeshMatrixMDVariableRefT}

| C++ Name                          | Element Type | Maximum Dimension |
|-----------------------------------|--------------|-------------------|
| \arcane{MeshMDVariableRefT}       | scalar       | 3                 |
| \arcane{MeshVectorMDVariableRefT} | vector       | 2                 |
| \arcane{MeshMatrixMDVariableRefT} | matrix       | 1                 |

These three variables have a common base class \arcane{MeshMDVariableRefBaseT}.

Usage is similar regardless of the element type. Access is done via the
`operator()` because it allows for multiple arguments. Starting from C++23, it
will also be possible to use the `operator[]`.

Before using these variables, it is necessary to call the
\arcane{MeshMDVariableRefT::reshape()} method. The number of values to specify
must be identical to the variable's dimension.

Example for scalar multi-dimensional variables:

\snippet MDVariableUnitTest.cc SampleMDVariableScalar

Example for vector multi-dimensional variables:

\snippet MDVariableUnitTest.cc SampleMDVariableVector

Example for matrix multi-dimensional variables:

\snippet MDVariableUnitTest.cc SampleMDVariableMatrix

Internally, all these variables are similar to a mesh array variable whose
number of elements is equal to the product of the components of the shape
returned by \arcane{MeshMDVariableRefBaseT::fullShape()}.
____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_entrypoint
</span>
</div>
