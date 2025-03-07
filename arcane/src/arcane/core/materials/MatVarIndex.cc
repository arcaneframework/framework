﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatVarIndex.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Index sur les variables matériaux.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatVarIndex.h"

#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const MatVarIndex& mvi)
{
  o << mvi.arrayIndex() << ":" << mvi.valueIndex();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
