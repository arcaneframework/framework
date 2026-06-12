# Synchronization Comparison {#arcanedoc_debug_perf_compare_synchronization}

Since version 3.11 of %Arcane, there is an automatic mechanism that allows
comparing variable values before and after a synchronization. This allows one to
know if a synchronization is useful or not.

\note Currently, this mechanism only works for simple synchronizations (those
called via the \arcane{MeshVariableRef::synchronize()} method.

To activate this mode, you must set the environment variable
`ARCANE_AUTO_COMPARE_SYNCHRONIZE`. The three possible values are:

- `1` : to activate the mechanism and display at the end of the calculation for
  each variable the number of synchronizations that modified the ghost cell
  values.
- `2` : like `1` but additionally there is a listing printout at the time of
  synchronization if the synchronization did not modify any values (which
  suggests that it is potentially not useful).
- `3` : like `2` but additionally the call stack at the time of synchronization
  is displayed.

Note that modes `2` and `3` require performing a reduction for each
synchronization, which can impact performance.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_unit_tests
</span>
</div>
