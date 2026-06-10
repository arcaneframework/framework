# Memory Pool {#arcanedoc_acceleratorapi_memorypool}

[TOC]

%Arcane has included a memory pool mechanism since version 3.14.10 (November
2024) that allows retaining a portion of the memory allocated for accelerators,
thereby avoiding costly calls to allocation or deallocation functions.

\note This mechanism is only functional for CUDA and ROCM/HIP.
Starting from version 4.1 of %Arcane, the memory pool manager is enabled by
default.

\warning Using the memory pool can change the code's behavior by removing
implicit synchronizations performed on the streams associated with allocations
and deallocations. Specifically, calls such as `cudaMalloc()` or `cudaFree()`.
The
page [CUDA implicit synchronization behavior and conditions in detail] (https://forums.developer.nvidia.com/t/cuda-implicit-synchronization-behavior-and-conditions-in-detail/251729)
explains this behavior for CUDA.

It is possible to enable and modify the behavior of the memory pool by setting
environment variables.

<table>
<tr><th>Environment Variable</th><th>Description</th></tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_POOL</td>
<td>
Indicates the type of memory for which the pool should be activated. The values
are specified by a combination of bits:
- 1 for managed memory (\arcane{eMemoryResource::UnifiedMemory})
- 2 for accelerator memory (\arcane{eMemoryResource::Device})
- 4 for host-pinned memory (\arcane{eMemoryResource::HostPinned})

If the environment variable value is `7`, for example, the memory pool is active
for these 3 types of memory resources.
If the value is `0`, the memory pool is disabled for all memories.
</td>
</tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_POOL_MAX_BLOCK_SIZE</td>
<td>
Indicates the maximum size (in bytes) of the blocks kept in the memory pool. A
high value allows for fewer allocations and deallocations but, in return,
retains more memory, which reduces the amount available for allocations that do
not go through the memory pool. The default value is 1MB (1024*1024).
</td>
</tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_PRINT_LEVEL</td>
<td>
Indicates whether memory usage information is displayed. This information is
useful for debugging only. Possible values are:
- 0 displays no information
- 1 displays usage statistics at the end of the calculation
- 2 same as 1 and displays information during reallocations
- 3 same as 2 and displays the call stack for a reallocation of unnamed arrays.
- 4 same as 3 but displays the call stack during reallocation for all arrays.
</td>
</tr>

</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi_reduction
</span>
</div>
