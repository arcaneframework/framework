// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableBuildInfo.h                                 (C) 2000-2012 */
/*                                                                           */
/* Référence à une variable sur un matériau du maillage.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATERIALVARIABLEBUILDINFO_H
#define ARCANE_MATERIALS_MATERIALVARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableBuildInfo.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT MaterialVariableBuildInfo
: public VariableBuildInfo
{
 public:

  MaterialVariableBuildInfo(IMeshMaterialMng* mng,const String& name,int property =0);
  MaterialVariableBuildInfo(IMeshMaterialMng* mng,const VariableBuildInfo& vbi);

 public:
  
  IMeshMaterialMng* materialMng() const { return m_material_mng; }
  
 private:

  IMeshMaterialMng* m_material_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

