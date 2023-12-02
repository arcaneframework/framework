// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.cc                                            (C) 2000-2023 */
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
  UniqueArray<Int32> local_ids[2];
  IMeshComponent* my_component = _component();
  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell ec = *ienvcell;
      if (ec.component()==my_component){
        MatVarIndex idx = ec._varIndex();
        if (idx.arrayIndex()==0){
          internals[0].add(ec._internal());
          matvar_indexes[0].add(idx);
          local_ids[0].add(ec.globalCell().localId());
        }
        else{
          internals[1].add(ec._internal());
          matvar_indexes[1].add(idx);
          local_ids[1].add(ec.globalCell().localId());
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
