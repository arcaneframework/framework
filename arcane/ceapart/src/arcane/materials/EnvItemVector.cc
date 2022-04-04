// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvCellVector.cc                                            (C) 2000-2016 */
/*                                                                           */
/* Vecteur sur les entités d'un milieu.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/EnvItemVector.h"
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

EnvCellVector::
EnvCellVector(const CellGroup& group,IMeshEnvironment* environment)
: ComponentItemVector(environment)
{
  _build(group.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvCellVector::
EnvCellVector(CellVectorView view,IMeshEnvironment* environment)
: ComponentItemVector(environment)
{
  _build(view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EnvCellVector::
_build(CellVectorView view)
{
  UniqueArray<ComponentItemInternal*> internals[2];
  UniqueArray<MatVarIndex> matvar_indexes[2];
  IMeshComponent* my_component = _component();
  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell ec = *ienvcell;
      if (ec.component()==my_component){
        MatVarIndex idx = ec._varIndex();
        if (idx.arrayIndex()==0){
          internals[0].add(ec.internal());
          matvar_indexes[0].add(idx);
        }
        else{
          internals[1].add(ec.internal());
          matvar_indexes[1].add(idx);
        }
      }
    }
  }
  this->_setItemsInternal(internals[0],internals[1]);
  this->_setMatVarIndexes(matvar_indexes[0],matvar_indexes[1]);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
