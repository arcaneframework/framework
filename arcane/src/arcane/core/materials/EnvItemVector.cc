// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Vecteur sur les entités d'un milieu.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/EnvItemVector.h"

#include "arcane/utils/FixedArray.h"

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
  FixedArray<UniqueArray<ConstituentItemIndex>,2> internals;
  FixedArray<UniqueArray<MatVarIndex>,2> matvar_indexes;
  FixedArray<UniqueArray<Int32>,2> local_ids;
  IMeshComponent* my_component = _component();
  ENUMERATE_ALLENVCELL(iallenvcell,_materialMng()->view(view)){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell ec = *ienvcell;
      if (ec.component()==my_component){
        MatVarIndex idx = ec._varIndex();
        if (idx.arrayIndex()==0){
          internals[0].add(ec._constituentItemIndex());
          matvar_indexes[0].add(idx);
          local_ids[0].add(ec.globalCell().localId());
        }
        else{
          internals[1].add(ec._constituentItemIndex());
          matvar_indexes[1].add(idx);
          local_ids[1].add(ec.globalCell().localId());
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
