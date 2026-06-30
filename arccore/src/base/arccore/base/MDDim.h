// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDDim.h                                                     (C) 2000-2026 */
/*                                                                           */
/* Tag for N-dimensional arrays.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MDDIM_H
#define ARCCORE_BASE_MDDIM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExtentsV.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constant for a rank 0 dynamic array
using MDDim0 = ExtentsV<Int32>;

//! Constant for a rank 1 dynamic array
using MDDim1 = ExtentsV<Int32, DynExtent>;

//! Constant for a rank 2 dynamic array
using MDDim2 = ExtentsV<Int32, DynExtent, DynExtent>;

//! Constant for a rank 3 dynamic array
using MDDim3 = ExtentsV<Int32, DynExtent, DynExtent, DynExtent>;

//! Constant for a rank 4 dynamic array
using MDDim4 = ExtentsV<Int32, DynExtent, DynExtent, DynExtent, DynExtent>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDDimType<0, IndexType_>
{
 public:

  using DimType = ExtentsV<IndexType_>;
};
template <typename IndexType_>
class MDDimType<1, IndexType_>
{
 public:

  using DimType = ExtentsV<IndexType_, DynExtent>;
};
template <typename IndexType_>
class MDDimType<2, IndexType_>
{
 public:

  using DimType = ExtentsV<IndexType_, DynExtent, DynExtent>;
};
template <typename IndexType_>
class MDDimType<3, IndexType_>
{
 public:

  using DimType = ExtentsV<IndexType_, DynExtent, DynExtent, DynExtent>;
};
template <typename IndexType_>
class MDDimType<4, IndexType_>
{
 public:

  using DimType = ExtentsV<IndexType_, DynExtent, DynExtent, DynExtent, DynExtent>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
