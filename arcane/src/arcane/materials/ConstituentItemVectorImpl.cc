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
, m_part_data(std::make_unique<MeshComponentPartData>(component, String()))
, m_recompute_part_data_functor(this, &ConstituentItemVectorImpl::_recomputePartData)
{
  Int32 level = -1;
  if (component->isMaterial())
    level = LEVEL_MATERIAL;
  else if (component->isEnvironment())
    level = LEVEL_ENVIRONMENT;
  else
    ARCANE_FATAL("Bad internal type of component");
  m_component_shared_info = m_material_mng->_internalApi()->componentItemSharedInfo(level);
  m_constituent_list = std::make_unique<ConstituentItemLocalIdList>(m_component_shared_info, String());
  m_part_data->setRecomputeFunctor(&m_recompute_part_data_functor);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_setItems(ConstArrayView<ConstituentItemIndex> globals,
          ConstArrayView<ConstituentItemIndex> multiples)
{
  m_constituent_list->copyPureAndPartial(globals, multiples);

  // TODO: Ne pas remettre à jour systématiquement les
  // 'm_items_local_id' mais ne le faire qu'à la demande
  // car ils ne sont pas utilisés souvent.
  Int32 nb_pure = globals.size();
  Int32 nb_impure = multiples.size();

  m_matvar_indexes.resize(nb_pure + nb_impure);
  m_items_local_id.resize(nb_pure + nb_impure);

  ComponentItemSharedInfo* cisi = m_component_shared_info;

  for (Int32 i = 0; i < nb_pure; ++i) {
    ConstituentItemIndex cii = globals[i];
    m_matvar_indexes[i] = cisi->_varIndex(cii);
    m_items_local_id[i] = cisi->_globalItemBase(cii).localId();
  }

  for (Int32 i = 0; i < nb_impure; ++i) {
    ConstituentItemIndex cii = multiples[i];
    m_matvar_indexes[nb_pure + i] = cisi->_varIndex(cii);
    m_items_local_id[nb_pure + i] = cisi->_globalItemBase(cii).localId();
  }

  // Mise à jour de MeshComponentPartData
  m_nb_pure = nb_pure;
  m_nb_impure = nb_impure;
  const bool do_lazy_evaluation = true;
  if (do_lazy_evaluation)
    m_part_data->setNeedRecompute();
  else
    _recomputePartData();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemVectorImpl::
_recomputePartData()
{
  // Mise à jour de MeshComponentPartData
  auto mvi_pure_view = m_matvar_indexes.subView(0, m_nb_pure);
  auto mvi_impure_view = m_matvar_indexes.subView(m_nb_pure, m_nb_impure);
  m_part_data->_setFromMatVarIndexes(mvi_pure_view, mvi_impure_view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ConstituentItemVectorImpl::
_view() const
{
  return { m_component, m_matvar_indexes,
           m_constituent_list->view(), m_items_local_id };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
