// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatItemVector.h"

#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/ConstituentItemVectorBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(const CellGroup& group, IMeshMaterial* material)
: MatCellVector(ConstituentItemVectorBuildInfo(group), material)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(CellVectorView view, IMeshMaterial* material)
: MatCellVector(ConstituentItemVectorBuildInfo(view), material)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(SmallSpan<const Int32> local_ids, IMeshMaterial* material)
: MatCellVector(ConstituentItemVectorBuildInfo(local_ids), material)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(const ConstituentItemVectorBuildInfo& build_info, IMeshMaterial* material)
: ComponentItemVector(material)
{
  _build(build_info._localIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatCellVector::
_build(SmallSpan<const Int32> local_ids)
{
  this->_setItems(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterial* MatCellVector::
material() const
{
  return static_cast<IMeshMaterial*>(component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
