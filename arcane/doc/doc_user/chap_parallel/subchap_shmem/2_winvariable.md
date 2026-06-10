# Variables in Shared Memory {#arcanedoc_parallel_shmem_winvariable}

[TOC]

\warning This feature is experimental. If variables with support are allocated
in shared memory, there may be significant adaptations required to make
collective resizing calls (risk of deadlocks). These resizings are slower than
in local memory. It is recommended to use shared memory only for variables
without support. It is not recommended to use shared memory with variables on
particles; calls to `endUpdate()` are rather frequent, and with shared memory,
they become collective!

## Introduction {#arcanedoc_parallel_shmem_winvariable_intro}

%Arcane variables usually use the default allocator to allocate memory. Without
a GPU, local machine memory is used, and with a GPU, unified memory is used.

A new memory allocator (internal to %Arcane) is available and allows memory to
be allocated in shared machine memory.
To do this, internally, we use the previously presented class:
Arcane::MachineShMemWin. We will therefore have access to non-contiguous
segments.

This mode is compatible with all %Arcane variable types (except scalar variables
without support (e.g., `VariableScalarReal`) and partial variables).

The main difficulty in using this shared memory mode is ensuring that all calls
that reallocate memory are collective.

For variables resized by %Arcane, the user does not need to worry about these
collective calls; %Arcane handles them. For example, with mesh variables,
%Arcane handles resizing if the mesh evolves.

\warning For variables with support, at the family level, using shared memory
variables makes calls acting on these variables collective. We can mention the
methods `Arcane::IItemFamily::compactItems()` and
`Arcane::IItemFamily::endUpdate()`.

Conversely, for variables for which a `resize()` method is available (or
`reshape()` for multi-dimensional variables), it is necessary to ensure that all
machine subdomains call this method (even if doing `var.resize(var.size())` for
subdomains that do not require resizing).

Setting aside these resizing calls, the use of shared memory variables is
identical to the use of local memory variables.

To declare a variable in shared memory, simply add the `IVariable::PInShMem`
property when creating it (in AXL files, the corresponding option is
`in-shmem="true"`).

## Accessing Memory Segments of Other Subdomains {#arcanedoc_parallel_shmem_winvariable_shared}

The utility of putting variables in shared memory is to be able to access data
from other subdomains without message exchanges.

To access data from all subdomains, you can use the `MachineShMemWinVariable`
classes. One class per %Arcane variable type:

<table>
  <tr>
    <th>Variable Type<br>(`example`)</th>
    <th>Class to Use</th>
  </tr>

  <tr>
    <td>1D Array Variable without support<br>(`Arcane::VariableArrayInt32`)</td>
    <td>Arcane::MachineShMemWinVariableArrayT</td>
  </tr>

  <tr>
    <td>Mesh Scalar Variable<br>(`Arcane::VariableCellInt32`)</td>
    <td>Arcane::MachineShMemWinMeshVariableScalarT</td>
  </tr>

  <tr>
    <td>2D Array Variable without support<br>(`Arcane::VariableArray2Int32`)</td>
    <td>Arcane::MachineShMemWinVariableArray2T</td>
  </tr>

  <tr>
    <td>Mesh 1D Array Variable<br>(`Arcane::VariableCellArrayInt32`)</td>
    <td>Arcane::MachineShMemWinMeshVariableArrayT</td>
  </tr>

  <tr>
    <td>Scalar Multi-dimensional Variable<br>(`Arcane::MeshMDVariableRefT<Cell, Real, MDDim2>`)</td>
    <td>Arcane::MachineShMemWinMeshMDVariableT</td>
  </tr>

  <tr>
    <td>Vector Multi-dimensional Variable<br>(`Arcane::MeshVectorMDVariableRefT<Cell, Real, 7, MDDim2>`)</td>
    <td>Arcane::MachineShMemWinMeshVectorMDVariableT</td>
  </tr>

  <tr>
    <td>Matrix Multi-dimensional Variable<br>(`Arcane::MeshMatrixMDVariableRefT<Cell, Real, 2, 5, MDDim1>`)</td>
    <td>Arcane::MachineShMemWinMeshMatrixMDVariableT</td>
  </tr>
</table>

Three methods are common to these classes:

- `machineRanks()`,
- `barrier()`,
- `updateVariable()`.

The first two have already been briefly described in the previous section
(\ref arcanedoc_parallel_shmem_winarray_var_usage).

`Arcane::MachineShMemWinVariableCommon::machineRanks()` allows retrieving the
ranks of the computation node's subdomains.

For example, if the returned view contains `[0, 2, 4, 6]`, we know that the
computation node possesses these subdomains and that we have access to their
data via `MachineShMemWin`.<br>
By using the `Arcane::IParallelMng::commSize()` method, knowing that the ranks
are contiguous, we can also determine which subdomains are not in our
computation node.
For example, if `commSize() = 8`, then the subdomains for which we must perform
inter-node communications are subdomains `[1, 3, 5, 7]`.

<br>

`Arcane::MachineShMemWinVariableCommon::barrier()` allows performing a barrier
for all subdomains of the computation node (so, if we take the previous example,
a barrier for subdomains `[0, 2, 4, 6]`).

This is useful in the case where subdomains use a memory window to share
information, to wait until each subdomain has written to its window before other
subdomains in the node read this data. The granularity is smaller than
`Arcane::IParallelMng::barrier()`.

<br>

The real difference from the previous section is the method
`Arcane::MachineShMemWinMeshVariableArrayT::updateVariable()`.

Internally, as explained in the introduction, we use an allocator that allocates
memory in shared memory and we use the `Arcane::MachineShMemWin` class to access
it.<br>
`Arcane::MachineShMemWinVariable` in turn uses `Arcane::MachineShMemWin` to
access the shared memory of the variables.

The problem is that the size of an array in %Arcane is not necessarily the same
size as the memory allocated by it. Consequently, internally, we cannot rely on
the size returned by `Arcane::MachineShMemWin` to build views on the variables.

We must therefore retrieve the sizes of the variables from each subdomain in
another way. To do this, we use a memory window to share them.

When changing the size of a variable (via a change in the mesh or via a resize
for array variables), we must update the variable sizes.

Today, **it is up to the user to do this via a call to `updateVariable()`**.

It is also possible to destroy the `Arcane::MachineShMemWinVariable` object and
recreate it after updating the variable.

### Examples {#arcanedoc_parallel_shmem_winvariable_shared_examples}

Some examples to illustrate the use of these classes:

<div style="text-align: center;">**Example 1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1

In this example, each subdomain has an array of two `Int32`.

\note The array could be of a different size for each subdomain.

Each subdomain puts its rank in the two cells of the array, and then each
subdomain displays the view of each array (`var_sh.view(rank)` returns a view of
two `Int32` from the `rank` array).

The call to the `updateVariable()` method could easily be removed by putting
`var.resize(2);` between the creation of the variable and the creation of the
`MachineShMemWinVariable`:

<div style="text-align: center;">**Example 1.1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1_1

An alternative to calling `updateVariable()` is the destruction/recreation of
`MachineShMemWinVariable`:

<div style="text-align: center;">**Example 1.2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1_2

----

<div style="text-align: center;">**Example 2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2

For mesh quantities, we have access to the operator
`Arcane::MachineShMemWinMeshVariableScalarT::operator()()` which allows
accessing the value of an `Item` using its `local_id`.

\warning The `local_id` is local to the target subdomain. It is therefore
necessary to share it in some way. You must not use the `local_id` of one
subdomain to access the `Items` of another subdomain!

If multiple values need to be read from another subdomain, it is strongly
recommended to do so by retrieving a view using the method
`Arcane::MachineShMemWinMeshVariableScalarT::view()`. Example:

<div style="text-align: center;">**Example 2.1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2_1

\warning If there have been deletions of `Items` without compaction, the code in
**Example 2.1** will display values of deleted `Items`.

In **Example 2**, the barrier is important, given that each subdomain will
access the data of the other subdomains.<br>
Nevertheless, it is also possible to do this to avoid the barrier:

<div style="text-align: center;">**Example 2.2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2_2

----

<div style="text-align: center;">**Example 3**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_3

Here, we have a 2D array without support.

\note As with the 1D array, the 2D array could be of a different size for each
subdomain.

The method `Arcane::MachineShMemWinVariableArray2T::view()` allows retrieving a
view (of type Arcane::Span2) on the 2D array of another subdomain of the
computation node.

----

<div style="text-align: center;">**Example 4**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_4

A mesh 1D array variable is a 2D array but with the first dimension
corresponding to the number of `Items`.

\warning Compared to the 2D array variable without support, the size of the
second dimension must be identical for each subdomain.

We find the method `Arcane::MachineShMemWinMeshVariableArrayT::view()`, which
returns a view of the 2D array of the variable from another subdomain.
The first dimension takes a `local_id` of an `Item` from the other subdomain and
the second dimension is the position in the array of the `Item`.

----

<div style="text-align: center;">**Example 5**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_5

With multi-dimensional variables, the method
`Arcane::MachineShMemWinMeshMDVariableT::view()` returns an `Arcane::MDSpan`
with one extra dimension compared to the variable's dimension, the first
dimension corresponding to the support.

The operator `Arcane::MachineShMemWinMeshMDVariableT::operator()()` is also
available and allows retrieving a multi-dimensional view of the variable's
dimension (since the `local_id` is also provided).

As mentioned previously, if accessing multiple `Items` arrays for a given
subdomain, it is better to retrieve a complete view via
`Arcane::MachineShMemWinMeshMDVariableT::view()`.

----

<div style="text-align: center;">**Example 6**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_6

For MD vector and matrix variables, we find the same methods as in the previous
example.

## Checkpoint {#arcanedoc_parallel_shmem_winvariable_checkpoints}

Shared memory variables are compatible with the checkpoint mechanism.

A variable property has been added to allow not saving the specified subdomain
arrays.

This is the `IVariable::PDumpNull` property. This is not a property reserved for
shared memory variables.

This property, when specified on a variable for a given subdomain, allows saving
an empty array. This is particularly useful in recovery for shared memory
variables given the obligation to perform collective operations.

### Examples {#arcanedoc_parallel_shmem_winvariable_checkpoints_examples}

<div style="text-align: center;">**Example 7**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_checkpoints_examples_7

We will call a master subdomain of the computation node, the subdomain with the
smallest rank of the node (since `machine_ranks` is sorted in ascending order,
it is the first rank of the array).

In this example, during the first iteration of the time loop, we resize the
variable for all subdomains.

Then, we assign the `IVariable::PDumpNull` property to all non-master
subdomains.

Finally, during recovery, we check that the master subdomains' arrays have been
restored and that the other arrays are empty.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_shmem_winarray
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
