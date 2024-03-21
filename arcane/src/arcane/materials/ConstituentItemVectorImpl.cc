// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorImpl.cc                                (C) 2000-2024 */
/*                                                                           */
/* Implémentation de 'IConstituentItemVectorImpl'.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ConstituentItemVectorImpl.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(IMeshComponent* component)
: m_material_mng(component->materialMng())
, m_component(component)
, m_matvar_indexes(platform::getDefaultDataAllocator())
, m_items_local_id(platform::getDefaultDataAllocator())
  , m_part_data(std::make_unique<MeshComponentPartData>(component,String()))
{
  Int32 level = -1;
  if (component->isMaterial())
    level = LEVEL_MATERIAL;
  else if (component->isEnvironment())
    level = LEVEL_ENVIRONMENT;
  else
    ARCANE_FATAL("Bad internal type of component");
  ComponentItemSharedInfo* shared_info = m_material_mng->_internalApi()->componentItemSharedInfo(level);
  m_constituent_list = std::make_unique<ConstituentItemLocalIdList>(shared_info,String());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(const ComponentItemVectorView& rhs)
: ConstituentItemVectorImpl(rhs.component())
{
  RunQueue& queue = m_material_mng->_internalApi()->runQueue();
  m_constituent_list->copy(rhs._constituentItemListView());
  m_matvar_indexes.copy(rhs._matvarIndexes());
  m_items_local_id.copy(rhs._internalLocalIds());
  m_part_data->_setFromMatVarIndexes(rhs._matvarIndexes(), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                  ConstArrayView<MatVarIndex> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_matvar_indexes.resize(nb_global + nb_multiple);

  m_matvar_indexes.subView(0, nb_global).copy(globals);
  m_matvar_indexes.subView(nb_global, nb_multiple).copy(multiples);

  {
    Int32Array& idx = m_part_data->_mutableValueIndexes(eMatPart::Pure);
    idx.resize(nb_global);
    for (Integer i = 0; i < nb_global; ++i)
      idx[i] = globals[i].valueIndex();
  }

  {
    Int32Array& idx = m_part_data->_mutableValueIndexes(eMatPart::Impure);
    idx.resize(nb_multiple);
    for (Integer i = 0; i < nb_multiple; ++i)
      idx[i] = multiples[i].valueIndex();
  }

  m_part_data->_notifyValueIndexesChanged(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_setLocalIds(ConstArrayView<Int32> globals, ConstArrayView<Int32> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_items_local_id.resize(nb_global + nb_multiple);

  m_items_local_id.subView(0, nb_global).copy(globals);
  m_items_local_id.subView(nb_global, nb_multiple).copy(multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ConstituentItemVectorImpl::
_view() const
{
  return ComponentItemVectorView(m_component, m_matvar_indexes,
                                 m_constituent_list->view(), m_items_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
