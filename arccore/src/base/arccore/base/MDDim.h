// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDDim.h                                                     (C) 2000-2025 */
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

template <>
class MDDimType<0>
{
 public:

  using DimType = MDDim0;
};
template <>
class MDDimType<1>
{
 public:

  using DimType = MDDim1;
};
template <>
class MDDimType<2>
{
 public:

  using DimType = MDDim2;
};
template <>
class MDDimType<3>
{
 public:

  using DimType = MDDim3;
};
template <>
class MDDimType<4>
{
 public:

  using DimType = MDDim4;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
