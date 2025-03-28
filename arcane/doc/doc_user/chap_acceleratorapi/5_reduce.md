# Réductions {#arcanedoc_acceleratorapi_reduction}

[TOC]

%Arcane permet d'effectuer sur accélérateur les trois types classiques
de réduction (`Min`, `Max` et `Somme`). Il existe deux possibilités
pour effectuer des réductions :

- via une boucle classique *RUNCOMMAND_* (\ref arcanedoc_acceleratorapi_reduceclass)
- via une api spécifique (\ref arcanedoc_acceleratorapi_reducedirect)

## Réductions via une boucle {#arcanedoc_acceleratorapi_reduceclass}

Les classes \arcaneacc{ReducerMax2}, \arcaneacc{ReducerMin2} et
\arcaneacc{ReducerSum2} permettent d'effectuer des réductions sur
accélérateurs. Elles s'utilisent à l'intérieur des boucles
telles que RUNCOMMAND_LOOP1() ou RUNCOMMAND_ENUMERATE() ou
RUNCOMMAND_MAT_ENUMERATE().

Il faut d'abord déclarer un instance d'une des classes de réduction
pui la passer en paramètre supplémentaires des boucles. Par exemple :

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

\warning Chaque instance ne peut être utilisée qu'une seule fois.

Il est possible d'utiliser plusieurs instances de réduction si on
souhaite réaliser plusieurs réductions à la fois. Par exemple:

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

## Réductions directes {#arcanedoc_acceleratorapi_reducedirect}

La classe \arcaneacc{GenericReducer} permet de lancer une commande
spécifique dédiée à la réduction. Une instance de
\arcaneacc{GenericReducer} peut être utilisée plusieurs fois.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi_materials
</span>
<span class="next_section_button">
\ref arcanedoc_acceleratorapi_memorypool
</span>
</div>
