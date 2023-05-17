// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2023 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryAllocator.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
_applySimdPaddingView(ArrayView<DataType> ids)
{
  const Integer size = ids.size();
  if (size==0)
    return;
  Integer padding_size = arcaneSizeWithPadding(size);
  if (padding_size==size)
    return;
  // Il faut utiliser directement le pointeur car sinon en mode check
  // cela fait un débordement de tableau.
  DataType* ptr = ids.unguardedBasePointer();
  DataType last_value = ptr[size-1];
  for( Integer k=size; k<padding_size; ++k )
    ptr[k] = last_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
_applySimdPadding(Array<DataType>& ids)
{
  const Integer size = ids.size();
  if (size==0)
    return;
  Integer padding_size = arcaneSizeWithPadding(size);
  if (padding_size==size)
    return;
  if (ids.allocator()->guarantedAlignment()<AlignedMemoryAllocator::simdAlignment())
    ARCANE_FATAL("Allocator guaranted alignment ({0}) has to be greated than {1}",
                 ids.allocator()->guarantedAlignment(),AlignedMemoryAllocator::simdAlignment());
  if (padding_size>ids.capacity())
    ARCANE_FATAL("Not enough capacity c={0} min_expected={1}",ids.capacity(),
                 padding_size);
  _applySimdPaddingView(ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void applySimdPadding(Array<Int16>& ids){ _applySimdPadding(ids); }
void applySimdPadding(Array<Int32>& ids){ _applySimdPadding(ids); }
void applySimdPadding(Array<Int64>& ids){ _applySimdPadding(ids); }
void applySimdPadding(Array<Real>& ids){ _applySimdPadding(ids); }

void applySimdPadding(ArrayView<Int16> ids){ _applySimdPaddingView(ids); }
void applySimdPadding(ArrayView<Int32> ids){ _applySimdPaddingView(ids); }
void applySimdPadding(ArrayView<Int64> ids){ _applySimdPaddingView(ids); }
void applySimdPadding(ArrayView<Real> ids){ _applySimdPaddingView(ids); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
