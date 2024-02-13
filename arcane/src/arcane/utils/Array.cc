// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2024 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/ArrayUtils.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class ArraySimdPadder
{
 public:

  static void applySimdPaddingView(ArrayView<DataType> ids)
  {
    const Int32 size = ids.size();
    if (size == 0)
      return;
    Int32 padding_size = arcaneSizeWithPadding(size);
    if (padding_size == size)
      return;

    // Construit une vue avec la taille du padding
    // pour éviter les débordement de tableau lors des tests
    ArrayView<DataType> padded_ids(padding_size, ids.data());

    DataType last_value = ids[size - 1];
    for (Int32 k = size; k < padding_size; ++k)
      padded_ids[k] = last_value;
  }

  static void applySimdPadding(Array<DataType>& ids)
  {
    const Integer size = ids.size();
    if (size == 0)
      return;
    Integer padding_size = arcaneSizeWithPadding(size);
    if (padding_size == size)
      return;
    MemoryAllocationArgs args;
    if (ids.allocator()->guarantedAlignment(args) < AlignedMemoryAllocator::simdAlignment())
      ARCANE_FATAL("Allocator guaranted alignment ({0}) has to be greated than {1}",
                   ids.allocator()->guarantedAlignment(args), AlignedMemoryAllocator::simdAlignment());
    if (padding_size > ids.capacity())
      ARCANE_FATAL("Not enough capacity c={0} min_expected={1}", ids.capacity(),
                   padding_size);
    applySimdPaddingView(ids.view());
  }

  static void checkSimdPadding(ConstArrayView<DataType> ids)
  {
    Integer size = ids.size();
    if (size == 0)
      return;
    Integer padding_size = arcaneSizeWithPadding(size);
    if (padding_size == size)
      return;

    // Construit une vue avec la taille du padding
    // pour éviter les débordement de tableau lors des tests
    ConstArrayView<DataType> padded_ids(padding_size, ids.data());

    // Vérifie que le padding est fait avec la dernière valeur valide.
    Int32 last_id = ids[size - 1];
    for (Integer k = size; k < padding_size; ++k)
      if (padded_ids[k] != last_id)
        ARCANE_FATAL("Bad padding value i={0} expected={1} value={2}",
                     k, last_id, padded_ids[k]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void applySimdPadding(Array<Int16>& ids)
{
  ArraySimdPadder<Int16>::applySimdPadding(ids);
}
void applySimdPadding(Array<Int32>& ids)
{
  ArraySimdPadder<Int32>::applySimdPadding(ids);
}
void applySimdPadding(Array<Int64>& ids)
{
  ArraySimdPadder<Int64>::applySimdPadding(ids);
}
void applySimdPadding(Array<Real>& ids)
{
  ArraySimdPadder<Real>::applySimdPadding(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void applySimdPadding(ArrayView<Int16> ids)
{
  ArraySimdPadder<Int16>::applySimdPaddingView(ids);
}
void applySimdPadding(ArrayView<Int32> ids)
{
  ArraySimdPadder<Int32>::applySimdPaddingView(ids);
}
void applySimdPadding(ArrayView<Int64> ids)
{
  ArraySimdPadder<Int64>::applySimdPaddingView(ids);
}
void applySimdPadding(ArrayView<Real> ids)
{
  ArraySimdPadder<Real>::applySimdPaddingView(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayUtils::
checkSimdPadding(ConstArrayView<Int32> ids)
{
  ArraySimdPadder<Int32>::checkSimdPadding(ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
