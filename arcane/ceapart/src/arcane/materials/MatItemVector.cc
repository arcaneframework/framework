// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatCellVector.cc                                            (C) 2000-2016 */
/*                                                                           */
/* Vecteur sur les entités d'un matériau.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MatItemVector.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  IMeshComponent* my_component = _component();

  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        if (mc.component()==my_component){
          MatVarIndex idx = mc._varIndex();
          if (idx.arrayIndex()==0){
            internals[0].add(mc.internal());
            matvar_indexes[0].add(idx);
          }
          else{
            internals[1].add(mc.internal());
            matvar_indexes[1].add(idx);
          }
        }
      }
    }
  }
  this->_setItemsInternal(internals[0],internals[1]);
  this->_setMatVarIndexes(matvar_indexes[0],matvar_indexes[1]);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
