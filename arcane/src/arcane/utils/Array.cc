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

#include "arcane/utils/ArraySimdPadder.h"
#include "arcane/utils/ArrayUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
