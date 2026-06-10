# Usage of the NumArray class {#arcanedoc_core_types_numarray}

[TOC]

The Arcane::NumArray class allows managing multi-dimensional arrays of numerical
values. The current version of %Arcane handles arrays up to dimension 4. The
number of dimensions of the array is also called the rank
(Arcane::NumArray::rank()) of the array.

This class is similar to the `std::mdarray` class planned for
C++26: https://isocpp.org/files/papers/D1684R0.html.

The semantics are by value (like `std::vector`), and therefore assignment
operators cause a copy of the array values.

The prototype is as follows:

~~~{cpp}
template<typename DataType,typename Extents,typename LayoutPolicy>
class NumArray;
~~~

With:
- \a DataType: the data type of the array. It must necessarily be a numeric
  type (`std::is_arithmetic<DataType>==true`) that must be trivially copyable
  (`std::is_trivially_copyable<DataType>==true`)
- \a Extents: indicates the number of elements (extent()) in each dimension. The
  value can be dynamic (Arcane::DynExtent) or static if a positive value is
  used.
- \a LayoutPolicy: indicates the layout policy. Currently, two values are
  possible: Arcane::RightLayout or Arcane::LeftLayout. The default value is
  Arcane::RightLayout, which corresponds to the classic layout of a
  multidimensional C array.

## Creation {#arcanedoc_core_types_numarray_creation}

The types Arcane::MDDim1, Arcane::MDDim2, Arcane::MDDim3, Arcane::MDDim4, allow
specifying instances whose dimensions are all dynamic. For example:

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarations

If you wish to specify one or more static dimensions, you can do so like this:

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsExtented

\note The instance values are not initialized during construction. You must call
the Arcane::NumArray::fill() method (only if the memory is accessible from the
host) if you wish to fill the array with a given value.

It is possible to specify the number of elements in each dimension during
construction or using the Arcane::NumArray::resize() method. In this case, the
number of arguments corresponds to the number of dynamic dimensions of the
instance:

\snippet NumArrayUnitTest.cc SampleNumArrayResize

\warning Resizing does not preserve the current values of the array

## Memory Management {#arcanedoc_core_types_numarray_memory_manager}

The Arcane::eMemoryRessource type allows specifying in which memory space the
array will be allocated. By default, Arcane::eMemoryRessource::UnifiedMemory is
used, which allows the array to be accessible both on the host and the
accelerator. It is possible to specify the associated memory resource during
construction. If you use the Arcane::eMemoryRessource::Device memory area, the
data will only be accessible on the accelerator, and you must not attempt to
access the array values (either for reading or writing) from the host.

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsMemory

## Indexing {#arcanedoc_core_types_numarray_indexing}

Indexing the values of Arcane::NumArray is done via the
Arcane::NumArray::operator(). You can either use an instance of
Arcane::ArrayIndex (`Arcane::ArrayIndex<N>` where `N` is the rank of the array)
or use an overload that takes `N` values as arguments.

For each dimension, the index value starts at zero. The valid values therefore
range from `[0,extentP()[` where `P` is the P-th dimension.

For example:

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsIndexation

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_array_usage
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span> -->
</div>
