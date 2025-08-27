// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableBuildInfo.h                                 (C) 2000-2025 */
/*                                                                           */
/* Informations pour une construire une variable matériau.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATERIALVARIABLEBUILDINFO_H
#define ARCANE_CORE_MATERIALS_MATERIALVARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableBuildInfo.h"

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MaterialVariableBuildInfo
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

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

