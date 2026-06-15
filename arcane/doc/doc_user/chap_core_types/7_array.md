# Arrays {#arcanedoc_core_types_array_usage}

[TOC]

## Array Types {#arcanedoc_core_types_array_usage_type}

\note Even if we refer to %Arcane in this section, the classes managing arrays
and views are in %Arccore and are therefore defined in the Arccore namespace.
Nevertheless, there is a `using` statement in %Arcane for these classes, which
allows them to be used as if they were in the Arcane namespace.

The use of arrays in %Arcane utilizes two types of classes: containers and
views:

- Array containers allow storing elements and manage the memory for this storage
  in the manner of `std::vector`. They possess operations allowing elements to
  be added or removed. The necessary memory is automatically managed when
  elements are added, for example.
- Views represent a subset of a container and are <strong>temporary</strong>
  objects: views should not be preserved between changes in the number of
  elements of the associated container.

The classes managing containers have a name that ends with \a %Array (for
example \arccore{UniqueArray} or \arccore{SharedArray}).
Containers have the following characteristics:

- they manage the necessary memory to store their elements.
- the elements are stored contiguously in memory. It is therefore possible to
  use these containers in C language functions, for example, that take pointers
  as arguments.

In %Arcane, there are two types of classes for managing views:
- classes whose name ends with \a View (\arccore{ArrayView},
  \arccore{ConstArrayView}. These are the classes historically used in %Arcane.
  For these classes, the number of elements is stored in an \arccore{Int32}.
- classes whose name ends with \a Span (\arccore{Span}, \arccore{SmallSpan}).
  These classes were added starting in 2018 and are similar to the `std::span`
  class (https://en.cppreference.com/w/cpp/container/span) in C++20. For these
  classes, the number of elements is stored in an \arccore{Int64} for
  \arccore{Span} and in an \arccore{Int32} for \arccore{SmallSpan}.

The major difference between historical views and \arccore{Span} is that there
is only one class to manage constant or non-constant elements, and the access
operators (\arccore{Span::operator[]()}) are `const` for `Span`, which allows
them to be used in lambdas.
It is therefore preferable to use \arccore{Span} or \arccore{SmallSpan} instead
of \arccore{ArrayView} or \arccore{ConstArrayView}.

Views have the following characteristics:
- they do not manage any memory and all originate from a container (which is not
  necessarily a %Arcane class)
- they are only valid as long as the associated container exists and the number
  of its elements is not modified.
- they are generally used by value rather than by reference (the & operator is
  not applied to them).
- their size is small (generally 16 bytes) and they can therefore be stored and
  copied easily.

\warning For performance reasons, array classes do not manage element
initialization in the same way if the type is considered a POD (Plain Old Data)
type for %Arcane.
The macro ARCCORE_DEFINE_ARRAY_PODTYPE(type) allows indicating that \a type is a
POD type for \arccore{Array}. The use of this macro must be done before defining
an array instance for the type \a type. All basic C++ types (`char`, `int`,
`double`, ...) are considered POD types for %Arcane.

The following table lists the classes managing arrays and their associated
views:

<table>
<tr>
<th>Description</th>
<th>Base Class</th>
<th>Reference Semantics</th>
<th>Value Semantics</th>
<th>Mutable View</th>
<th>Const View</th>
</tr>
<tr>
<td>1D Array</td>
<td>\arccore{Array}</td>
<td>\arccore{SharedArray}</td>
<td>\arccore{UniqueArray}</td>
<td>\arccore{ArrayView} <br/> \arccore{Span<T>} <br/> \arccore{SmallSpan<T>}</td>
<td>\arccore{ConstArrayView} <br/> \arccore{Span<const T>} <br/> \arccore{SmallSpan<const T>}</td></tr>
<tr>
<td>Classic 2D Array</td>
<td>\arccore{Array2}</td>
<td>\arccore{SharedArray2}</td>
<td>\arccore{UniqueArray2}</td>
<td>\arccore{Array2View} <br/> \arccore{Span2<T>} <br/> \arccore{SmallSpan2<T>}</td>
<td>\arccore{ConstArray2View} <br/> \arccore{Span2<const T>} <br/> \arccore{SmallSpan2<const T>}</td>
</tr>
<tr>
<td>2D Array with variable 2nd dimension</td>
<td>\arcane{MultiArray2}</td>
<td>\arcane{SharedMultiArray2}</td>
<td>\arcane{UniqueMultiArray2}</td>
<td>\arcane{MultiArray2View}</td>
<td>\arcane{ConstMultiArray2View}</td>
</tr>
</table>

For each array type, there is a base class from which an implementation with
reference semantics and an implementation with value semantics inherit. The base
class is neither copyable nor assignable. The difference in semantics concerns
the operation of copy and assignment operators:
- reference semantics means that when you do <em>a = b</em>, \a a becomes a
  reference to \a b, and any modification to \a b also modifies \a a.

```cpp
Arcane::SharedArray<int> a1(5);
Arcane::SharedArray<int> a2;
a2 = a1; // a2 and a1 refer to the same memory area.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // prints '2'
```

- value semantics means that when you do <em>a = b</em>, \a a becomes a copy of
  the values of \a b, and subsequently the arrays \a a and \a b are independent.

```cpp
Arcane::UniqueArray<int> a1(5);
Arcane::UniqueArray<int> a2;
a2 = a1; // a2 becomes a copy of a1.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // prints '1'
```

## Passing Arrays as Arguments {#arcanedoc_core_types_array_usage_argument}

Here are the best practices to follow when passing arrays as arguments:

<table>

<tr>
<th>Argument</th>
<th>Need</th>
<th>Possible Operations</th>
</tr>
<tr>
<td>\arccore{ConstArrayView} <br/> \arccore{Span<const T>} <br/> \arccore{SmallSpan<const T>}</td>
<td>1D array read-only</td>
<td>

```cpp
x = a[i];
```

</td>
</tr>
<tr>
<td>\arccore{ArrayView} <br/> \arccore{Span<T>} <br/> \arccore{SmallSpan<T>}</td>
<td>1D array read and/or write, but size is not modifiable</td> 
<td>

```cpp
x = a[i];
a[i] = y;
```

</td>
</tr>
<tr>
<td>\arccore{Array}&</td>
<td>1D array modifiable and capable of changing element count</td>
<td>

```cpp
x = a[i];
a[i] = y;
a.resize(u);
a.add(v);
```

</td>
</tr>
<tr>
<td>const \arccore{Array}&</td>
<td>Forbidden. Use \arccore{ConstArrayView} or \arccore{Span<const T>} instead</td>
<td></td>
</tr>
<tr>
<td>\arccore{ConstArray2View} <br/> \arccore{Span2<const T>} <br/> \arccore{SmallSpan2<const T>}</td>
<td>2D array read-only</td>
<td>

```cpp
x = a[i][j];
```

</td>
</tr>
<tr>
<td>\arccore{Array2View} <br/> \arccore{Span2<T>} <br/> \arccore{SmallSpan2<T>}</td>
<td>2D array read and/or write, but size is not modifiable</td>
<td>

```cpp
x = a[i][j];
a[i][j] = y;
```

</td>
</tr>
<tr>
<td>\arccore{Array}&</td>
<td>2D array modifiable and capable of changing element count</td>
<td>

```cpp
x = a[i][j];
a[i][j] = y;
a.resize(u,v);
```

</td>
</tr>
<tr>
<td>const \arccore{Array2}&</td>
<td>Forbidden. Use \arccore{ConstArray2View} or \arccore{Span2<const T>} instead</td>
<td></td>
</tr>
</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_timeloop
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_numarray
</span>
</div>
