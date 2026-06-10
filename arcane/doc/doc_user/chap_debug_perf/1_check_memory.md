# Memory Problem Detection {#arcanedoc_debug_perf_check_memory}

Arcane has a mechanism to detect certain memory problems, particularly:
- memory leaks
- deallocations that do not correspond to any allocation.

Furthermore, this allows for statistics on memory usage.

\warning This mechanism currently only works on Linux OS.

\warning This mechanism does not work when multi-threading is enabled.

To activate it, simply set the environment variable ARCANE_CHECK_MEMORY to
\c true. All allocations and deallocations are traced. However, for performance
reasons, the call stack is only preserved and displayed for allocations
exceeding a certain size. By default, the value is 1MB (1000000). It is possible
to specify another value via the environment variable
ARCANE_CHECK_MEMORY_BLOCK_SIZE. The environment variable
ARCANE_CHECK_MEMORY_BLOCK_SIZE_ITERATION allows specifying a block value that
will be used for the loop after initialization. This allows for finer tracing of
allocations during computation than those that occur during initialization.

Calls are traced from the call to ArcaneMain::arcaneInitialize() up to the call
to ArcaneMain::arcaneFinalize(). During this last call, a list of allocated
blocks that have not been deallocated is displayed.

It is possible to manage the memory checker more finely via the IMemoryInfo
interface. This interface is a singleton accessible via the method
arcaneGlobalMemoryInfo();

\note INTERNAL: For now, any inconsistencies between allocations and
deallocations are indicated on std::cout. This can cause readability issues in
parallel. In the future, ITraceMng will need to be used, but this is tricky
currently because this mechanism also performs memory calls and it is difficult
to make it compatible with the current debugging functions.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
</div>
