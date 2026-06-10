# Shared Memory Arrays {#arcanedoc_parallel_shmem_winarray}

[TOC]

## Introduction {#arcanedoc_parallel_shmem_winarray_intro}

Two different implementations are available: one implementation with all
segments contiguous and with a constant size, defined when the object is
constructed, and another implementation with non-contiguous segments and a
variable size.

## Implementation with Contiguous Segments {#arcanedoc_parallel_shmem_winarray_const}

This implementation allows the creation of a memory window whose segments are
all contiguous. It is thus quite simple to re-slice the segments during use (for
example, to balance a calculation).

### Usage {#arcanedoc_parallel_shmem_winarray_const_usage}

This part is managed by the Arcane::ContigMachineShMemWin class.

This class can use three implementations of IContigMachineShMemWinBase, one for
each type of Arcane::IParallelMng.
It is therefore possible to use this class regardless of whether you have an
MpiParallelMng, a SequentialParallelMng, a SharedMemoryParallelMng, or a
HybridParallelMng (\ref arcanedoc_execution_launcher_exchange).

The creation of an object of this type is collective. An instance of this class
will create a memory window composed of several segments (one per subdomain).

Access to the elements of the segments is not collective. Concurrent access to
an element is possible using semaphores, mutexes, or std::atomic.
For std::atomic, the operations must be `address-free`:

```c
bool is_lock_free = std::atomic<Real>{}.is_lock_free();
```

When this object is constructed, each subdomain provides a segment size. The
window size will be equal to the sum of the segment sizes.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_1

\remark It is possible that MPI does not support shared memory (if using MPICH
in ch3:sock mode, for example). To check this, you can use the function
ParallelMngUtils::isMachineShMemWinAvailable(IParallelMng* pm).

To access its segment, you can use the method
Arcane::ContigMachineShMemWin::segmentView().

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_3

Once the segment has been modified, you can perform a barrier to ensure that
everyone has written to their segment before using it.

To find out which subdomains share a window on the node, you can retrieve an
array of ranks.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_2

The position of the ranks in this array corresponds to the position of their
segment in the window.

To read the segments of other subdomains on the node, you can use the method
Arcane::ContigMachineShMemWin::segmentConstView().

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_4

The window size cannot be modified. However, the implementation in %Arcane
allows resizing the segments collectively (provided that the new window size is
less than or equal to the original size).

\note An implementation with memory windows having non-contiguous segments could
be more performant but would make this functionality impossible.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_5

\remark The elements of the window are not modified during resizing.

Since the window is contiguous, access to the entire window is possible for all
subdomains.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_6

## Implementation with Non-Contiguous Segments {#arcanedoc_parallel_shmem_winarray_var}

This implementation is quite different from the previous one.
Here, the segments of the memory windows are no longer contiguous. Furthermore,
with this implementation, it is possible to resize the segments like a classic
dynamic array.

Nevertheless, this operation is collective, which contaminates most of the
implementation's methods.

### Usage {#arcanedoc_parallel_shmem_winarray_var_usage}

This part is managed by the Arcane::MachineShMemWin class.

As with the previous implementation, this one is compatible with all %Arcane
parallelism modes.

The creation of an object of this type is collective. An instance of this class
will create a memory window composed of several segments (one per subdomain).

Like a UniqueArray, it is possible to specify an initial size (here `5`):
\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_7

And it is possible not to specify an initial size.
\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_8

The method Arcane::MachineShMemWin::machineRanks() is available and returns the
same array as the Arcane::ContigMachineShMemWin implementation.

To explore our segment or the segment of another subdomain, you can use the same
methods as before:

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_9

However, since the segments are not contiguous, the `windowView()` methods are
not available.

The segments have a size that can be increased or decreased over time.

It is possible to add elements using the method
Arcane::MachineShMemWin::add(Arcane::Span<const Type> elem):

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_10

This method is collective; all subdomains on a node must call it. If a subdomain
does not wish to add elements to its segment, it can call the `add()` method
with an empty array or without arguments (Arcane::MachineShMemWin::add()).

This operation can be costly due to memory reallocation. It is therefore
advisable to add a large quantity of elements at once rather than element by
element.

If element-by-element addition is indispensable, the method
Arcane::MachineShMemWin::reserve(Arcane::Int64 new_capacity) is available to
avoid reallocating a segment multiple times:

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_11

In this piece of code, we will reserve space for `20` `Integer`s for all
subdomains. This value can be different for each subdomain (if a subdomain does
not want to reserve more space, it can call Arcane::MachineShMemWin::reserve()).

\note With this method, you cannot reserve less space than already reserved
(calling `reserve(0)` has no effect). To reduce the reserved space, the method
Arcane::MachineShMemWin::shrink() is available.

\warning As with UniqueArray, the method
Arcane::MachineShMemWin::reserve(Arcane::Int64 new_capacity) does not have the
same function as the method
Arcane::MachineShMemWin::resize(Arcane::Int64 new_nb_elem). The former only
reserves memory space, but this space remains inaccessible without `add()` or
without `resize()`. The latter changes the number of elements in the segment and
calls `reserve()` if necessary.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_12

In our example, `resize()` will increase the number of elements in all segments
except for the subdomain that previously performed the `add()` operations (which
has 15 elements, compared to 5 for the others).
This subdomain will go from 15 elements to 12.

Like the `reserve()` method, each subdomain can set the value it wants.

It is also possible to add elements to the segment of another subdomain using
the collective method Arcane::MachineShMemWin::addToAnotherSegment(Arcane::Int32
rank, Arcane::Span<const Type> elem).

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_13


\warning It is impossible to mix calls to `add()` and `addToAnotherSegment()`.
If a subdomain calls the `addToAnotherSegment()` method, all subdomains must
collectively call `addToAnotherSegment()` (with or without parameters) and not
`add()`.

The functionality is almost identical to the `add()` method but with an extra
parameter to designate the rank of the subdomain possessing the segment to be
modified.

\warning Two subdomains cannot add elements to the same segment (at the same
time) (which prevents concurrency issues).
```c
// Not possible:
if (my_rank == 0){
  window.addToAnotherSegment();
}
else if (my_rank == 1){
  window.addToAnotherSegment(0, mon_tableau);
}
else if (my_rank == 2){
  window.addToAnotherSegment(0, mon_tableau);
}
```
```c
// Possible:
if (my_rank == 0){
  window.addToAnotherSegment();
  window.addToAnotherSegment();
}
else if (my_rank == 1){
  window.addToAnotherSegment();
  window.addToAnotherSegment(0, mon_tableau);
}
else if (my_rank == 2){
  window.addToAnotherSegment(0, mon_tableau);
  window.addToAnotherSegment();
}
```

## Shared Memory Between Processes {#arcanedoc_parallel_shmem_winarray_shmem}

Inter-process shared memory should not be seen as multithreaded shared memory.
This sharing only occurs on a part of the memory, not on all of the memory.

Consider this structure:

```c
struct MaStruct
{
    MaStruct()
    : array_integer(10)
    {}
    
    UniqueArray<Integer> array_integer;
};
```

It can be used like this:

```c
MaStruct ma_struct;
ma_struct.array_integer[0] = 123 * (my_rank+1);
```

If this structure is used in a window, it would look like this:

```c
ContigMachineShMemWin<MaStruct> win_struct(pm, 1);
Span<MaStruct> my_span = win_struct.segmentView();
new (my_span.data()) MaStruct();

my_span[0].array_integer[0] = 123 * (my_rank+1);
```

You can display the value you assigned, and it works correctly:

```c
debug() << "Elem : " << my_span[0].array_integer[0];
```

But if you want to display the value of another process:

```c
window.barrier();

Span<MaStruct> other_span = win_struct.segmentView(machine_ranks[(my_rank + 1) % machine_nb_proc]);
debug() << "Elem : " << other_span[0].array_integer[0];

my_span[0].~MaStruct();
```

In multi-process mode (launching with `mpirun -n 2 ...`), the program will
crash (segfault), whereas it will not crash in multithreading (launching with
`-A,S=2`).

In multi-process mode, the attributes of the
`UniqueArray<Integer> array_integer;` array in the structure are not allocated
in shared memory (the `new` or `malloc` calls are made in local memory), so
other processes do not have access to them.

It is also important to note that the same memory location in shared memory is
addressed differently between processes. Therefore, if you provide a shared
memory allocator to the `UniqueArray`, the addresses used will only be valid
locally.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_shmem
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_shmem_winvariable
</span>
</div>
