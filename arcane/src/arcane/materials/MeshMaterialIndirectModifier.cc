// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialIndirectModifier.cc                             (C) 2000-2022 */
/*                                                                           */
/* Objet permettant de modifier indirectement les matériaux.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ItemGroup.h"
#include "arcane/IItemFamily.h"

#include "arcane/materials/MeshMaterialIndirectModifier.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialBackup.h"

#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialIndirectModifier::
MeshMaterialIndirectModifier(IMeshMaterialMng* mm)
: m_material_mng(mm)
, m_backup(new MeshMaterialBackup(mm,true))
, m_has_update(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialIndirectModifier::
~MeshMaterialIndirectModifier() noexcept(false)
{
  endUpdate();
  delete m_backup;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialIndirectModifier::
_endUpdate(bool do_sort)
{
  if (!m_has_update)
    return;

  if (do_sort){
    for( MeshMaterialVariableIndexer* v : m_material_mng->variablesIndexer() ){
      CellGroup cells = v->cells();
      UniqueArray<Int32> items_lid(cells.view().localIds());
      cells.clear();
      cells.setItems(items_lid,true);
    }
  }

  m_material_mng->forceRecompute();

  m_backup->restoreValues();

  m_has_update = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialIndirectModifier::
endUpdate()
{
  _endUpdate(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialIndirectModifier::
endUpdateWithSort()
{
  _endUpdate(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialIndirectModifier::
beginUpdate()
{
  if (m_has_update)
    ARCANE_FATAL("beginUpdate() already called.");
  m_has_update = true;
  m_backup->saveValues();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
