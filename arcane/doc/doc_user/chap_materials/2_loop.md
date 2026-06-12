# Loops over material and environment entities {#arcanedoc_materials_loop}

[TOC]

This page describes the management of loops over material and environment
entities.

In the rest of this page, the generic term \a component is used to describe a
material or an environment.

A component's entities can be divided into two parts: pure entities and impure
entities. By definition, entities that are not pure are impure. The notion of
purity varies depending on the component type:
- for an environment, an entity is pure if there is only one environment in that
  entity.
- for a material, an entity is pure if there is only one material <b>AND</b>
  only one environment.

At the memory storage level for a given variable, accessing a pure entity is
equivalent to accessing the global value of that variable.

## Loop Generalization {#arcanedoc_materials_loop_loop}

Since version 2.7.0 of %Arcane, the generic macro ENUMERATE_COMPONENTITEM()
allows iteration over a component's entities globally or by part (pure/impure).
It can replace the macros ENUMERATE_COMPONENTCELL(), ENUMERATE_MATCELL(), and
ENUMERATE_ENVCELL().

The following values are available for iteration:

ENUMERATE_COMPONENTITEM(MatCell,icell,container) with container of type
IMeshMaterial* or MatCellVector.


It is possible to iterate only over the pure or impure part of a component.

\note Currently, the traversal order of loops by pure or impure part is not
defined and may evolve later. This means that if there are dependencies between
loop iterations, the result may vary from one execution to another.

The following examples show the different variants of the
ENUMERATE_COMPONENTITEM() macro

### Loops over environments {#arcanedoc_materials_loop_envloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemEnv

### Loops over materials {#arcanedoc_materials_loop_matloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemMat

### Generic loops over components {#arcanedoc_materials_loop_componentloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemComponent

## Vector loops over components {#arcanedoc_materials_loop_simdloop}

\note In the current version of %Arcane (2.7.0), vector loops are only supported
for environments (but not yet for materials).

To be able to use vectorization on components, you must include the following
file:

```cpp
#include <arcane/materials/ComponentSimd.h>

using namespace Arcane::Materials;
```

It is necessary to use the C++11 lambda mechanism to iterate over components via
vector iterators. This is done using the following macro:

```cpp
ENUMERATE_COMPONENTITEM_LAMBDA(){
};
```

\warning Do not forget the final semicolon ';'. For more information, refer to
the documentation for this macro.

\note This mechanism is experimental and may evolve later.

For example, with the following variable declarations:

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateVariableDeclaration

It is possible to use vector loops as follows:

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateSimdComponentItem

\warning For performance reasons, the order of iterations may be arbitrary. It
is therefore essential that there are no relationships between the iterations.
In particular, if non-associative operations such as sums on real numbers are
used, the result may vary between two executions.

\note The current implementation has several limitations:
- It is not yet possible to use these enumerators with concurrent loops (see
  page \ref arcanedoc_materials_manage_concurrency).
- For SIMD, views must be used.
- For now, views are only available for scalar variables.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_materials_manage
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_materials_loop
</span> -->
</div>
