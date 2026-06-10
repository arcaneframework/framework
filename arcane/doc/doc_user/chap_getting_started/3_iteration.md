# Iteration {#arcanedoc_getting_started_iteration}

[TOC]

Before being able to code an operation, you must understand how a loop over a
list of mesh entities, such as cells or nodes, is written. Indeed, almost all
operations performed are done on a set of entities and therefore involve a loop
over a list of entities. For example, calculating the mass of the cells consists
of looping over all the cells and performing the product of its volume by its
density for each one. Conventionally, this can be written as follows:

```cpp
for( Integer i=0; i<nbCell(); ++i )
  m_cell_mass[i] = m_density[i] * m_volume[i];
```

The *for* loop consists of three parts separated by a semicolon. The first is
the initialization, the second is the loop exit test, and the third is the
operation performed between two iterations.

The previous writing has several disadvantages:
- it exposes the underlying data structure, namely an array;
- it uses an integer type index to access elements. This weak typing is a source
  of error because it does not allow, among other things, to account for the
  type of the variable. For example, one could write `m_velocity[i]` with `i`
  being a cell number and `m_velocity` being a node variable;
- it requires that the numbering of the entities be contiguous.

Considering that the list of entities is always traversed in the same order, it
is possible to model the previous behavior using four operations:

- initialize a counter at the beginning of the array;
- increment the counter;
- check if the counter is at the end of the array;
- return the element corresponding to the counter.

The mechanism is then general and independent of the container type: the set of
entities could be implemented as an array or a list without changing this
formalism. In the architecture, the counter above is called an *iterator* and
iterating over the set of elements is done by providing a start and end
iterator, otherwise called an *enumerator*.

In %Arcane, this enumerator derives from the base class \arcane{ItemEnumerator}
and has the following methods:

- a constructor taking a mesh entity group as an argument;
- *operator++()*: to access the next element;
- *hasNext()*: to test if we are at the end of the iteration;
- _operator*(): which returns the current element.

To add an additional level of abstraction and to allow code instrumentation,
%Arcane provides a function in the form of a macro for each enumerator type. It
is therefore not necessary to use the methods of \arcane{ItemEnumerator}. This
function has the following prototype:

```cpp
ENUMERATE_(kind, nom_iterateur, nom_groupe )
```

with:
- **kind** the type of the entity (\arcane{Node}, \arcane{Cell}, ...),
- **nom_iterateur** the name of the iterator
- **nom_groupe** the name of the group (\arcane{ItemGroup}) over which we
  iterate.

When you are in a module (whose base class is \arcane{BasicModule}) or a
service (whose base class is \arcane{BasicService}), %Arcane provides methods to
access the group containing all entities of a given entity type. For example,
the method \arcane{BasicModule::allCells()} allows retrieving the group of all
cells. Thus, to iterate over all cells, with **i** as the name of the iterator,
you can do this:

```cpp
ENUMERATE_(Cell,i,allCells())
```

The mass calculation loop described previously then becomes:

```cpp
ENUMERATE_(Cell,i,allCells()){
  m_cell_mass[i] = m_density[i] * m_volume[i];
}
```

The type of an enumerator depends on the type of the mesh element: an enumerator
over a group of nodes is not the same type as an enumerator over a group of
cells, and they are therefore incompatible. For example, if velocity is a node
variable, the following example causes a compilation error:

```cpp
cout << m_velocity[i]; // Erreur!
```

Similarly, it is impossible to write:

```cpp
ENUMERATE_(Cell,i,allNodes()) // Erreur!
```

because \arcane{BasicModule::allNodes()} is a group of nodes and **i** is an
enumerator over a group of cells.

Note that the enumerator's '*' operator allows access to the current element:
```cpp
ENUMERATE_(Cell,icell,allCells()){
  Cell cell = *icell;
}
```

It is possible to use the entity itself to retrieve the value of a variable, but
for performance reasons, you must prioritize access via the iterator:
```cpp
ENUMERATE_(Cell,icell,allCells()){
  Cell cell = *icell;
  m_cell_mass[cell] = m_density[cell] * m_volume[cell]; // less performant
  m_cell_mass[icell] = m_density[icell] * m_volume[icell]; // more performant
}
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started_basicstruct
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
