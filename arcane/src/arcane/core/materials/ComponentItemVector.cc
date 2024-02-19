// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVector.cc                                      (C) 2000-2024 */
/*                                                                           */
/* Vecteur sur les entités d'un composant.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemVector.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/MeshComponentPartData.h"
#include "arcane/core/materials/internal/ConstituentItemLocalIdList.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::Impl::
Impl(IMeshComponent* component)
: m_material_mng(component->materialMng())
, m_component(component)
, m_matvar_indexes(platform::getDefaultDataAllocator())
, m_items_local_id(platform::getDefaultDataAllocator())
, m_part_data(std::make_unique<MeshComponentPartData>(component))
{
  Int32 level = -1;
  if (component->isMaterial())
    level = LEVEL_MATERIAL;
  else if (component->isEnvironment())
    level = LEVEL_ENVIRONMENT;
  else
    ARCANE_FATAL("Bad internal type of component");
  ComponentItemSharedInfo* shared_info = m_material_mng->_internalApi()->componentItemSharedInfo(level);
  m_constituent_list = std::make_unique<ConstituentItemLocalIdList>(shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::Impl::
Impl(IMeshComponent* component, const ConstituentItemLocalIdListView& constituent_list_view,
     ConstArrayView<MatVarIndex> matvar_indexes, ConstArrayView<Int32> items_local_id)
: Impl(component)
{
  m_constituent_list->copy(constituent_list_view);
  m_matvar_indexes.copy(matvar_indexes);
  m_items_local_id.copy(items_local_id);
  m_part_data->_setFromMatVarIndexes(matvar_indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::Impl::
deleteMe()
{
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(IMeshComponent* component)
: m_p(new Impl(component))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(ComponentItemVectorView rhs)
: m_p(new Impl(rhs.component(), rhs._constituentItemListView(),
               rhs._matvarIndexes(), rhs._internalLocalIds()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setItems(ConstArrayView<ConstituentItemIndex> globals,
          ConstArrayView<ConstituentItemIndex> multiples)
{
  m_p->m_constituent_list->copyPureAndPartial(globals, multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                  ConstArrayView<MatVarIndex> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_p->m_matvar_indexes.resize(nb_global + nb_multiple);

  m_p->m_matvar_indexes.subView(0, nb_global).copy(globals);
  m_p->m_matvar_indexes.subView(nb_global, nb_multiple).copy(multiples);

  {
    Int32Array& idx = m_p->m_part_data->_mutableValueIndexes(eMatPart::Pure);
    idx.resize(nb_global);
    for (Integer i = 0; i < nb_global; ++i)
      idx[i] = globals[i].valueIndex();
  }

  {
    Int32Array& idx = m_p->m_part_data->_mutableValueIndexes(eMatPart::Impure);
    idx.resize(nb_multiple);
    for (Integer i = 0; i < nb_multiple; ++i)
      idx[i] = multiples[i].valueIndex();
  }

  m_p->m_part_data->_notifyValueIndexesChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setLocalIds(ConstArrayView<Int32> globals, ConstArrayView<Int32> multiples)
{
  Integer nb_global = globals.size();
  Integer nb_multiple = multiples.size();

  m_p->m_items_local_id.resize(nb_global + nb_multiple);

  m_p->m_items_local_id.subView(0, nb_global).copy(globals);
  m_p->m_items_local_id.subView(nb_global, nb_multiple).copy(multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ComponentItemVector::
view() const
{
  return ComponentItemVectorView(m_p->m_component, m_p->m_matvar_indexes,
                                 m_p->m_constituent_list->view(), m_p->m_items_local_id);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView ComponentItemVector::
pureItems() const
{
  return m_p->m_part_data->pureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView ComponentItemVector::
impureItems() const
{
  return m_p->m_part_data->impureView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemLocalIdListView ComponentItemVector::
_constituentItemListView() const
{
  return m_p->m_constituent_list->view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
