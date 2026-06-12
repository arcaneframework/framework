# Vectorization {#arcanedoc_parallel_simd}

[TOC]

<!-- describes the use of vectorization (SIMD). -->

## Introduction {#arcanedoc_parallel_simd_intro}

Vectorization is a mechanism that allows the same instruction to be executed on
multiple data items. The English term commonly used to describe vectorization is
<strong >Single Instruction Multiple Data (SIMD)</strong>. Since this is an
instruction managed directly by the processor, the possible operations are quite
limited. Generally, these are basic arithmetic operations (addition,
subtraction, ...) as well as classic mathematical functions (minimum, maximum,
absolute value, ...). Complex mathematical operations (such as logarithm,
exponential, ...) are generally not native vector instructions.

Recent processors all support vectorization. However, the vector sizes and
possible operations differ from one processor to another.

For example, the following simple loop performs \b n additions:

```cpp
using namespace Arcane;
UniqueArray<Real> a, b, c;
for( int i=0; i<n; ++i ){
  a[i] = b[i] + c[i];
}
```

With a scalar processor, the registers contain only one
real number, and addition instructions therefore operate on only one
real number. \b n addition instructions will be required to perform this
calculation. A vector processor has registers containing
multiple real numbers. For registers containing \b P real numbers, the number
of addition instructions needed is therefore \b n/P. If scalar and vector
instructions take the same amount of time, there is therefore a theoretical
speedup factor of \b P. The larger the registers, the more important the
potential benefit of vectorization is. Of course, in practice, it is often less
rosy, and the real gain depends on other factors such as memory bandwidth,
pipelining, etc.

To exploit vectorization, there are two
possibilities (which are compatible):

- letting the compiler handle vectorization.
- using specific C++ classes designed for this purpose.

The first solution is the simplest because it does not require changing the
code. It is directly available via the correct compiler options. The downside of
this simplicity is that it is often difficult for the compiler to detect where
vectorization is possible. The generated code is therefore rarely vectorized.
The second method guarantees the exploitation of vectorization but requires
rewriting the code. %Arcane provides a set of classes to exploit this second
method.

The principle is therefore to provide a vector class corresponding
to a scalar class. The vector class will therefore contain \a N
scalar values, with \a N depending on the available vectorization type.

Even if theoretically vectorization can be applied to all simple types (short,
int, long, float, ...), %Arcane limits itself to providing classes that manage
vectorization only for the Arcane::Real and derived types (Arcane::Real2,
Arcane::Real3).

Currently, %Arcane provides the following vector types:

<table>
<tr>
<th>Scalar type</th><th>Vector type</th><th>Definition file</th>
</tr>
<tr>
<td>Arcane::Real</td><td>Arcane::SimdReal</td><td>

```cpp
#include "arcane/utils/Simd.h"
```
</td>
<tr>
<td>Arcane::Real2</td>
<td>Arcane::SimdReal2</td>
<td></td>
</tr>
<tr>
<td>Arcane::Real3</td>
<td>Arcane::SimdReal3</td>
<td></td>
</tr>
<tr>
<td>Arcane::Item, Arcane::Cell, Arcane::Face, ...</td>
<td>Arcane::SimdItem, Arcane::SimdCell, Arcane::SimdFace, ...</td>
<td>

```cpp
#include "arcane/SimdItem.h"
```
</td>
</tr>
</table>

\note For now, the Real2x2 and Real3x3 classes do not have an associated vector
class, but this will be available in a later version of %Arcane.

## Using Vector Classes {#arcanedoc_parallel_simd_usage}

Using SIMD classes is similar to scalar usage. It is generally sufficient to
change the name of the scalar classes to the corresponding vector name.

\note The use of vectorization assumes the use of views on variables. It is not
possible to access a variable directly via the Arcane::SimdItem and derived
classes.

The following example shows how to transition from scalar to vector writing:

```cpp
using namespace Arcane;

// Variable declaration
VariableCellReal pressure = ...;
VariableCellReal density = ...;
VariableCellReal adiabatic_cst = ...;
VariableCellReal internal_energy = ...;
VariableCellReal sound_speed = ...;

// Input views (reading)
// In C++11, it is also possible to use the 'auto' keyword:
// auto in_pressure = viewIn(pressure);
VariableCellRealInView in_pressure = viewIn(pressure);
VariableCellRealInView in_density = viewIn(m_density);
VariableCellRealInView in_adiabatic_cst = viewIn(adiabatic_cst);

// Output views (writing)
VariableCellRealOutView out_internal_energy = viewOut(internal_energy);
VariableCellRealOutView out_sound_speed = viewOut(sound_speed);

// Scalar version
ENUMERATE_CELL(icell,allCells()){
  Cell vi = *icell;
  Real pressure = in_pressure[vi];
  Real adiabatic_cst = in_adiabatic_cst[vi];
  Real density = in_density[vi];
  out_internal_energy[vi] = pressure / ((adiabatic_cst-1.0) * density);
  out_sound_speed[vi] = math::sqrt(adiabatic_cst*pressure/density);
}

// Vector version
ENUMERATE_SIMD_CELL(icell,allCells()){
  SimdCell vi = *icell;
  SimdReal pressure = in_pressure[vi];
  SimdReal adiabatic_cst = in_adiabatic_cst[vi];
  SimdReal density = in_density[vi];
  out_internal_energy[vi] = pressure / ((adiabatic_cst-1.0) * density);
  out_sound_speed[vi] = math::sqrt(adiabatic_cst*pressure/density);
}
```

Vectorization works well as long as all elements of the
vector must perform the same operation. Things get complicated when this is no
longer the case. Notably, anything that depends on a condition is difficult to
vectorize. There are also cases where you want to perform specific operations
for each element within a vector loop. To handle this situation, it is possible
to add sequential sections by iterating over the entities of an Arcane::SimdItem
using the ENUMERATE_*. macros. For example:

```cpp
using namespace Arcane;
ENUMERATE_SIMD_CELL(ivcell,allCells()){
  SimdCell simd_cell = *ivcell; // Vector of cells
  ENUMERATE_CELL(icell,ivcell){
    Cell cell = *icell;
    info() << "Cell: local_id=" << cell.localId();
  }
}
```

Finally, it is possible to know the number of reals in a vector register via the
SimdReal::BLOCK_SIZE constant. This allows, for example, iterating over the
elements of a vector register:

```cpp
using namespace Arcane;
SimdReal3 vr;
for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
  Real3 r = vr[i];
  info() << " R [" << i << "] = " << r;
}
```

## Alignment Management {#arcanedoc_parallel_simd_alignment}

In general, and this is the case for x64 processors, using
vectorization requires that the data in memory be aligned in a more restrictive
way than for scalar types. For SSE, AVX, and AVX512, the minimum alignment is
equal to the byte size of the Simd vector. So, for example, for AVX with 256-bit
vectors, which is 32 bytes, the minimum alignment is 32 bytes. To simplify
vectorization, %Arcane
guarantees that the following types have the desired minimum alignment for
vectorization:
- the localIds() of Arcane::ItemGroup.
- the data of array and scalar variables on the mesh.
- the data of 2D array variables and array variables on the mesh. Note that
  for the latter, the start of the array is aligned, but if the first dimension
  is not a multiple of the vector size, the subsequent elements will not be
  aligned because there is no padding management yet).

Since C++ does not allow allocation via new/delete with alignment,
%Arcane provides the Arccore::AlignedMemoryAllocator class which can be used
with the Arcane::UniqueArray and Arcane::SharedArray classes to guarantee
alignment. For example:

```cpp
using namespace Arcane;
UniqueArray x(AlignedMemoryAllocator::Simd());
x.resize(32); // Guaranteed that \a x has correct alignment.
```

## Loop End Management {#arcanedoc_parallel_simd_endloop}

Vectorization works well when the number of loop elements is a multiple of the
Simd vector size. If this is not the case, the last part of the loop must be
handled in a certain way. <strong>To provide an identical mechanism for all
vectorization types, %Arcane duplicates the last valid value in the Simd
vector</strong>.
For example, suppose the following code:

```cpp
using namespace Arcane;
CellGroup cells = ...
ENUMERATE_SIMD_CELL(ivcell,cells){
  SimdCell simd_cell = *ivcell; // Vector of cells
}
```

With \a cells being a group of cells containing 11 elements. If we assume the
vector size is 8, then the previous loop will perform two iterations. For the
first, we will have the following values for \a simd_cell

```cpp
// First iteration
simd_cell[0]  = cells[0];
simd_cell[1]  = cells[1];
simd_cell[2]  = cells[2];
simd_cell[3]  = cells[3];
simd_cell[4]  = cells[4];
simd_cell[5]  = cells[5];
simd_cell[6]  = cells[6];
simd_cell[7]  = cells[7];
```

For the second iteration, since \a cells only contains 11
elements, we repeat the last valid value in \a simd_cell:

```cpp
// Second iteration
simd_cell[8]  = cells[8];
simd_cell[9]  = cells[9];
simd_cell[10] = cells[10];
simd_cell[11] = cells[10]; // Repeats the last valid value.
simd_cell[12] = cells[10];
simd_cell[13] = cells[10];
simd_cell[14] = cells[10];
simd_cell[15] = cells[10];
```

This mechanism works partially as long as the operations
performed are truly vectorizable. If this is not the case, it is
possible to iterate only over the valid values as follows:

```cpp
using namespace Arcane;
CellGroup cells = ...
ENUMERATE_SIMD_CELL(ivcell,cells){
  SimdCell simd_cell = *ivcell; // Vector of cells
  ENUMERATE_CELL(icell,ivcell){
    Cell cell = *icell;
    info() << "Cell: local_id=" << cell.localId();
  }
}
```

With the previous example, the inner loop will only perform 3 iterations,
(for the cells \a cells[8], \a cells[9] and \a cells[10]) for the
last part of \a cells.

## Supported Operations {#arcanedoc_parallel_simd_operation}

The mathematical operations supported by %Arcane's vector classes are defined in
the SimdMathUtils.h file:

```cpp
#include "arcane/SimdMathUtils.h"
```

%Arcane provides for the Arcane::SimdReal, Arcane::SimdReal2, and
Arcane::SimdReal3 vector classes the same operations
available in MathUtils.h for the scalar version, with the exception of \a min
and \a max.

## Supported Vectorization Mechanisms {#arcanedoc_parallel_simd_support}

In version 2.2, %Arcane only supports vectorization for
x86 architecture processors.

For these processors, there are (currently) three generations of vectorization:

- SSE vectorization, which is available on all 64-bit processors and uses
  128-bit registers.
- AVX vectorization, which is available on processors
  since the SandyBridge generation (roughly since 2012). These vectors have a
  size of 256 bits.
- AVX512 vectorization, which is available on SkyLake generation processors
  (2015+) and has 512-bit vectors. This vectorization has been supported since
  version 2.3.9 of %Arcane.

Depending on the platform, several mechanisms may be
available. On Intel processors, processors have backward compatibility, so those
that support AVX512 also support AVX and SSE. Similarly, processors with AVX
support SSE.

%Arcane defines the default mechanism as the one that uses the most advanced
vectorization. The Arcane::SimdInfo, Arcane::SimdReal, Arcane::SimdReal3 types
are therefore typedefs that depend on the platform.

%Arcane also defines macros indicating the available mechanisms:

- ARCANE_HAS_SSE if SSE vectorization is available
- ARCANE_HAS_AVX if AVX or AVX2 vectorization is available
- ARCANE_HAS_AVX512 if AVX512 vectorization is available.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_concurrency
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_loadbalance
</span>
</div>
