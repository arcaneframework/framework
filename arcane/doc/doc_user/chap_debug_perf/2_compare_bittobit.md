# Bit-by-bit Comparison {#arcanedoc_debug_perf_compare_bittobit}

[TOC]

This page describes the tools available in ARCANE to perform bit-by-bit
comparisons of the values of variables managed by ARCANE. The following
comparisons are possible:
- \ref arcanedoc_debug_perf_compare_bittobit_two_exec
- \ref arcanedoc_debug_perf_compare_bittobit_synchronization
- \ref arcanedoc_debug_perf_compare_bittobit_replica

It is important to note that comparison is only possible on variables managed by
%Arcane (such as #VariableCellReal, #VariableScalarInt32, ...). In the current
implementation, only variables with a numeric data type are compared (so, for
example, not by the 'String' data types).

## Bit-by-bit Comparison Between Two Executions {#arcanedoc_debug_perf_compare_bittobit_two_exec}

This mechanism allows determining the list of %Arcane variables that are
different between two executions. It only works on %Arcane variables.

The operating principle is as follows:
- Execution of a reference case, with saving of the results.
- Execution of a second case and comparison during execution with the reference
  case.

\note Since a reference case must first be executed and the results saved to
disk, the saved data can be very voluminous depending on the cases.

\note Before version 1.22.2 of %Arcane, only variables on mesh entities were
compared. Since this version, array variables that do not rely on a mesh entity
are also compared.

All %Arcane variables are compared except those that have the
IVariable::PExecutionDepend property set to true. In parallel/sequential
comparison, those that also have the IVariable::PSubDomainDepend property are
not compared.

### Reference Execution {#arcanedoc_debug_perf_compare_bittobit_exec}

To save the reference information, simply run the case after setting the
environment variable **STDENV_VERIF** to the value **WRITE**. In this case, the
values of all variables will be saved at each iteration. It is possible to do
this for each entry point by setting the environment variable
**STDENV_VERIF_ENTRYPOINT**, but this greatly increases the volume of
information to save.

By default, the information is saved in the directory \c /tmp/$USER/verif, but
it is possible to change this by specifying another path in the environment
variable **STDENV_VERIF_PATH**

\note In the case of a comparison for a parallel case, you must be certain that
the path used for the data is accessible by all nodes of the computer, which is
generally not the case for the \c /tmp directory.

Since version 3.16 of %Arcane, it is possible to choose the comparison method
used to calculate the difference. This is done via the environment variable
**STDENV_VERIF_DIFF_METHOD**. The possible values are:

- `RELATIVE`: calculates the relative difference `(v-ref) / ref`. This is the
  comparison used by default.
- `LOCALNORMMAX` calculates the difference `(v-ref) / max_ref` where `max_ref`
  is the absolute value of the maximum reference values on the subdomain. Note
  that with this method, the difference depends on the decomposition.

### Comparison with Reference {#arcanedoc_debug_perf_compare_bittobit_compare}

Once the reference has been executed, simply set the environment variable
**STDENV_VERIF** to `READ` and run a new execution. It is possible to change the
number of subdomains compared to the reference execution and thus perform
comparisons between parallel and sequential (in this case, the sequential
execution must be the reference and be executed first) or parallel/parallel
comparisons.

In the execution listing, the list of variables that are different between this
execution and the reference, along with their values, will then appear for each
iteration (or for each entry point if **STDENV_VERIF_ENTRYPOINT** is defined),
as follows:

```log
*I-TimeLoopMng Processor 3 : 50 entité(s) ayant des valeurs différentes pour la variable CaracteristicLength:
50 entité(s) ayant des valeurs différentes pour la variable CaracteristicLength
VDIFF: Variable 'CaracteristicLength' (G) uid=1264 lid=673 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1263 lid=672 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1250 lid=659 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1272 lid=681 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1251 lid=660 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1252 lid=661 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1264 lid=39 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1263 lid=38 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1250 lid=25 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1272 lid=47 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
```

For each variable and each subdomain, its name, the Item::uniqueId() and the
Item::localId() of the entity, as well as whether it is ghost ((G)) or belongs
to the subdomain ((O)), the current value (val), the reference value (ref), and
the relative difference (rdiff) are indicated. To avoid cluttering the listing,
only the 10 most significant differences in absolute value are displayed.

When the variable is an array variable and is not on a mesh entity, (G) or (O)
does not appear, and instead of the local entity number, the index of the
element in the array is displayed.

In parallel, it may be normal for the values on ghost cells to be different
from the reference if the variable is not synchronized. Since this can be the
case for many variables, it is possible to display differences only on cells
belonging to the subdomain by setting the environment variable
**STDENV_VERIF_SKIP_GHOSTS**.

## Verification of Comparisons {#arcanedoc_debug_perf_compare_bittobit_verification}

### Synchronization Checks {#arcanedoc_debug_perf_compare_bittobit_synchronization}

Just as it is possible to perform bit-by-bit comparisons, it is possible to
verify that variables are properly synchronized between subdomains. To do this,
simply specify the value **CHECKSYNC** to the environment variable
**STDENV_VERIF**. Values with the *IVariable::PNoNeedSync* attribute and partial
variables are not compared.

### Value Checks Between Replicas {#arcanedoc_debug_perf_compare_bittobit_replica}

It is also possible to verify that the values of a variable are the same on all
replicas of a subdomain. To do this, you must specify the value **CHECKREPLICA**
to the environment variable **STDENV_VERIF**. Variables with the
*IVariable::PNoReplicaSync* attribute and partial variables are not compared.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_check_memory
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_compare_synchronization
</span>
</div>
