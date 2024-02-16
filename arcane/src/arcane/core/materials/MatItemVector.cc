// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatItemVector.h"

#include "arcane/utils/FixedArray.h"

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
  FixedArray<UniqueArray<ComponentItemInternalLocalId>,2> internals;
  FixedArray<UniqueArray<MatVarIndex>,2> matvar_indexes;
  FixedArray<UniqueArray<Int32>,2> local_ids;
  IMeshComponent* my_component = _component();

  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        if (mc.component()==my_component){
          MatVarIndex idx = mc._varIndex();
          if (idx.arrayIndex()==0){
            internals[0].add(mc._internalLocalId());
            matvar_indexes[0].add(idx);
            local_ids[0].add(mc.globalCell().localId());
          }
          else{
            internals[1].add(mc._internalLocalId());
            matvar_indexes[1].add(idx);
            local_ids[1].add(mc.globalCell().localId());
          }
        }
      }
    }
  }
  this->_setItems(internals[0],internals[1]);
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
