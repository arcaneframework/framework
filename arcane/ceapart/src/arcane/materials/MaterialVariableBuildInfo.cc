// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.cc                                  (C) 2000-2012 */
/*                                                                           */
/* Référence à une variable sur un matériau du maillage.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"

#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialVariableBuildInfo::
MaterialVariableBuildInfo(IMeshMaterialMng* mng,const String& name,int property)
: VariableBuildInfo(mng->mesh(),name,property)
, m_material_mng(mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialVariableBuildInfo::
MaterialVariableBuildInfo(IMeshMaterialMng* mng,const VariableBuildInfo& vbi)
: VariableBuildInfo(vbi)
, m_material_mng(mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
