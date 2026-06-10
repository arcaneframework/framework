# Parallelizing code {#arcanedoc_parallel}

If you wish to speed up your code, this chapter should interest you.
Several methods are available in %Arcane to allow code acceleration: using all
available CPU cores, using CPU vector units, and using accelerators (GPU).
In the case of unbalanced code, it is also possible to use load balancing, in
order to distribute the computational load equally across all subdomains.

<br>

Table of Contents for this chapter:

1. \subpage arcanedoc_parallel_intro <br>
   Introduction to parallelism introduced in %Arcane.

2. \subpage arcanedoc_parallel_concurrency <br>
   Presents the use of multi-threading in %Arcane (in addition to domain
   decomposition).

3. \subpage arcanedoc_parallel_simd <br>
   Presents the mechanisms available in %Arcane to use today's CPU vector units.

4. \subpage arcanedoc_parallel_loadbalance <br>
   Describes the use of the load balancing mechanism on the mesh.

5. \subpage arcanedoc_parallel_shmem <br>
   Describes the use of memory windows in shared memory.

____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_parallel_intro
</span>
</div>
