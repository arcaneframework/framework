# Reductions {#arcanedoc_acceleratorapi_reduction}

[TOC]

%Arcane allows performing the three classic types of reduction (`Min`, `Max`,
and `Sum`) on the accelerator. There are two ways to perform reductions:

- via a classic loop *RUNCOMMAND_* (\ref arcanedoc_acceleratorapi_reduceclass)
- via a specific API (\ref arcanedoc_acceleratorapi_reducedirect)

## Reductions via a loop {#arcanedoc_acceleratorapi_reduceclass}

The classes \arcaneacc{ReducerMax2}, \arcaneacc{ReducerMin2}, and
\arcaneacc{ReducerSum2} allow performing reductions on accelerators. They are
used inside loops such as RUNCOMMAND_LOOP1() or RUNCOMMAND_ENUMERATE() or
RUNCOMMAND_MAT_ENUMERATE().

First, you must declare an instance of one of the reduction classes and then
pass it as an additional parameter to the loops. For example:

```cpp
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/Reduce.h"
{
  Arcane::Accelerator::RunQueue queue = ...;
  auto command = makeCommand(queue);
  Arcane::Accelerator::ReducerMin2<double> minimum_reducer(command);
  Arcane::VariableCellReal my_variable = ...;
  auto in_my_variable = viewIn(command,my_variable);
  command << RUNCOMMAND_ENUMERATE(Cell,cid,allCells(),minimum_reducer)
  {
    minimum_reducer.combine(in_my_variable[cid]);
  };
  info() << "MinValue=" << minimum_reducer.reducedValue();
}
```

\warning Each instance can only be used once.

It is possible to use multiple reduction instances if you want to perform
several reductions at once. For example:

```cpp
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/Reduce.h"
{
  Arcane::Accelerator::RunQueue queue = ...;
  auto command = makeCommand(queue);
  Arcane::Accelerator::ReducerMin2<double> minimum_reducer(command);
  Arcane::Accelerator::ReducerMax2<double> maximum_reducer(command);
  Arcane::VariableCellReal my_variable = ...;
  auto in_my_variable = viewIn(command,my_variable);
  command << RUNCOMMAND_ENUMERATE(Cell,cid,allCells(),minimum_reducer,maximum_reducer)
  {
    minimum_reducer.combine(in_my_variable[cid]);
    maximum_reducer.combine(in_my_variable[cid]);
  };
  info() << "MinValue=" << minimum_reducer.reducedValue();
  info() << "MaxValue=" << maximum_reducer.reducedValue();
}
```

## Direct Reductions {#arcanedoc_acceleratorapi_reducedirect}

The \arcaneacc{GenericReducer} class allows launching a specific command
dedicated to reduction. An instance of \arcaneacc{GenericReducer} can be used
multiple times.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi_materials
</span>
<span class="next_section_button">
\ref arcanedoc_acceleratorapi_memorypool
</span>
</div>
