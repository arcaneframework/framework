// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Vecteur sur les entités d'un milieu.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/EnvItemVector.h"

#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellVector::
EnvCellVector(const CellGroup& group,IMeshEnvironment* environment)
: EnvCellVector(group.view().localIds(), environment)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellVector::
EnvCellVector(CellVectorView view,IMeshEnvironment* environment)
: EnvCellVector(view.localIds(), environment)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellVector::
EnvCellVector(SmallSpan<const Int32> local_ids, IMeshEnvironment* environment)
: ComponentItemVector(environment)
{
  _build(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnvCellVector::
_build(SmallSpan<const Int32> local_ids)
{
  this->_setItems(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshEnvironment* EnvCellVector::
environment() const
{
  return static_cast<IMeshEnvironment*>(component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
