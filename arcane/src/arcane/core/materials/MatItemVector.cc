// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.cc                                            (C) 2000-2023 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatItemVector.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(const CellGroup& group,IMeshMaterial* material)
: ComponentItemVector(material)
{
  _build(group.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatCellVector::
MatCellVector(CellVectorView view,IMeshMaterial* material)
: ComponentItemVector(material)
{
  _build(view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MatCellVector::
_build(CellVectorView view)
{
  UniqueArray<ComponentItemInternal*> internals[2];
  UniqueArray<MatVarIndex> matvar_indexes[2];
  UniqueArray<Int32> local_ids[2];
  IMeshComponent* my_component = _component();

  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        if (mc.component()==my_component){
          MatVarIndex idx = mc._varIndex();
          if (idx.arrayIndex()==0){
            internals[0].add(mc._internal());
            matvar_indexes[0].add(idx);
            local_ids[0].add(mc.globalCell().localId());
          }
          else{
            internals[1].add(mc._internal());
            matvar_indexes[1].add(idx);
            local_ids[1].add(mc.globalCell().localId());
          }
        }
      }
    }
  }
  this->_setItemsInternal(internals[0],internals[1]);
  this->_setMatVarIndexes(matvar_indexes[0],matvar_indexes[1]);
  this->_setLocalIds(local_ids[0],local_ids[1]);
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
