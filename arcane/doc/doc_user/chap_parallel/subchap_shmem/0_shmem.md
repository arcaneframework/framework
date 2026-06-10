# Shared Memory Windows in Multi-Process {#arcanedoc_parallel_shmem}

[TOC]

This section will describe how to use shared memory between processes on the
same computing node using memory windows.

A memory window is a memory space allocated in a portion of memory accessible by
all processes.
This window will be divided into several segments, one per process.

Two ways to exploit these windows are available:
- via arrays and views (as one would use Arcane::UniqueArray),
- via %Arcane variables.

From now on, we will refer to a computing node as a machine.

<br>

Table of Contents for this subsection:

1. \subpage arcanedoc_parallel_shmem_winarray <br>
   Presents the classes that allow shared memory to be used as %Arcane arrays.

2. \subpage arcanedoc_parallel_shmem_winvariable <br>
   Presents how to create and use %Arcane variables in shared memory.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_loadbalance
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_shmem_winarray
</span>
</div>
