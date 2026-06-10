# Integration with CUPTI (Cuda Profiling Tools Interface) {#arcanedoc_debug_perf_cupti}

[TOC]

## Description

[CUPTI](https://docs.nvidia.com/cupti/index.html) is a library provided by
NVIDIA. It allows, among other things, retrieving events concerning unified
memory management. This is the context in which %Arcane uses CUPTI.

CUPTI is used via environment variables

<table>
<tr><th>Environment Variable</th><th>Description</th></tr>

<tr>
<td>ARCANE_CUPTI_LEVEL</td>
<td>
Indicates the events we want to trace. Note that for level 2, exclusive access
to the GPU is required, so this mode does not work in parallel. Possible values
are:
- 0 inactive
- 1 unified memory transfers
- 2 same as 1 + compute kernels
</td>
</tr>

<tr>
<td>ARCANE_CUPTI_FLUSH</td>
<td>
Indicates when the event information is displayed. To have precise tracking, it
is necessary to display the information after each GPU kernel execution, but
this mode can significantly increase execution time. Possible values are:
- 0 no explicit flush
- 1 flush after each kernel
</td>
</tr>

<tr>
<td>ARCANE_CUPTI_PRINT</td>
<td>
Indicates whether we want to perform a display for each event. This can
considerably slow down execution time. Possible values are:
- 0 no display
- 1 listing display (on std::cout)
</td>
</tr>

<tr>
<td>ARCANE_CUDA_MALLOC_TRACE</td>
<td>
Indicates whether we want to trace all calls to `cudaMallocManaged()`. Possible
values are:
- 0 no trace
- 1 trace the array name
- 2 same as 1 + display of malloc and free
- 3 same as 2 + call stack
</td>
</tr>

<tr>
<td>ARCANE_CUDA_UM_PAGE_ALLOC</td>
<td>
Indicates the manner of allocation via `cudaMallocManaged()`. It is possible to
allocate a multiple of the memory page size for each allocation. Since unified
memory transfers happen page by page, this allows for better distinction of
which memory access caused the transfer. The trade-off is that each allocation
requires allocating at least one page (usually 4KB). Possible values are:
- 0 normal allocation
- 1 allocation by multiple of the page size.
</td>
</tr>

</table>

## Example

The following example allows tracing unified memory transfers and displaying the
name of the associated array.

```
ARCANE_CUDA_MALLOC_TRACE=1 ARCANE_CUPTI_FLUSH=1 ARCANE_CUPTI_LEVEL=1 ./my_test -A,AcceleratorRuntime=cuda toto.arc
```

With the following result

```
*I-ArcaneMasterInternal *** ITERATION       17  TIME 3.594972986357219e-03  LOOP       17  DELTAT 4.177248169415655e-04 ***
*I-ArcaneMasterInternal Date: 2023-10-26T09:40:35 Conso=(R=2.168,I=0.131,C=0.166) Mem=(222,m=222:0,M=222:0,avg=222)
UNIFIED_MEMORY_COUNTER [ 4179074172 4179078748 ] address=0x7f1cc01f9000 kind=BYTES_TRANSFER_HTOD value=24576 flags=3 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179078748 4179081788 ] address=0x7f1cc01ff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179241052 4179244924 ] address=0x7f1cc01f9000 kind=BYTES_TRANSFER_DTOH value=24576 flags=3 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179244924 4179246172 ] address=0x7f1cc01ff000 kind=BYTES_TRANSFER_DTOH value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4223957312 4223960384 ] address=0x7f1cc0bff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistory_Iterations_1 stack=
*I-ArcaneMasterInternal  
*I-ArcaneMasterInternal *** ITERATION       18  TIME 4.054470284992941e-03  LOOP       18  DELTAT 4.594972986357221e-04 ***
*I-ArcaneMasterInternal Date: 2023-10-26T09:40:35 Conso=(R=2.299,I=0.131,C=0.176) Mem=(222,m=222:0,M=222:0,avg=222)
UNIFIED_MEMORY_COUNTER [ 4353892054 4353895094 ] address=0x7f1cc0bff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistory_Iterations_1 stack=
```

The two values after `UNIFIED_MEMORY_COUNTER` correspond to the start and end
time of the transfer. The other fields are:
- `address`: memory address of the array
- `kind`: transfer type (`Host to device` or `Device to host`)
- `value`: quantity (in bytes) of memory transferred
- `flags`: if `2`, the transfer is explicitly requested by the code. If `3`, it
  is a speculative transfer initiated by the NVIDIA driver.
- `source` and `destination`: device number
- `name`: Arcane array name. This is only active if the environment variable
  `ARCANE_CUDA_MALLOC_TRACE` is at least 1. If the transfer is not linked to an
  Arcane array (%Arcane (\arccore{UniqueArray} or \arcane{NumArray})), there
  will be no associated name. Note that since transfers happen page by page, it
  is possible that the indicated array is not the one that caused the transfer.
  To avoid this effect, it is possible to allocate a page for each allocation by
  setting the environment variable `ARCANE_CUDA_UM_PAGE_ALLOC` to `1`.

At the end of the calculation, the total amount of memory transferred and the
number of transfers are displayed. For example:

```
MemoryTransferSTATS: HTOD = 17895424 (680) DTOH = 7102464 (301)
```

In this example, 680 transfers were made from the CPU to the GPU for 17MB of
data transferred. 301 transfers were made from the GPU to the CPU for 7MB of
data transferred.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_unit_tests
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling
</span>
</div>
