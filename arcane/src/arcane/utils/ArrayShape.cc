// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayShape.cc                                               (C) 2000-2022 */
/*                                                                           */
/* Représente la forme d'un tableau.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayShape.h"

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CheckedConvert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayShape::
ArrayShape(Span<const Int32> v)
{
  _set(v.smallView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayShape::
setNbDimension(Int32 nb_dim)
{
  if (nb_dim<0 || nb_dim>=MAX_NB_DIMENSION)
    ARCANE_THROW(ArgumentException,"Bad value for argument 'nb_value'");
  m_nb_dim = nb_dim;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayShape::
_set(SmallSpan<const Int32> v)
{
  Int32 vsize = v.size();
  if (vsize >= MAX_NB_DIMENSION)
    ARCANE_FATAL("Bad size '{0}' for shape. Maximum size is {1}",vsize,MAX_NB_DIMENSION);
  m_nb_dim = CheckedConvert::toInt32(vsize);
  for (Int32 i = 0; i < vsize; ++i)
    m_dims[i] = v[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayShape::
setDimensions(Span<const Int32> dims)
{
  _set(dims.smallView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
