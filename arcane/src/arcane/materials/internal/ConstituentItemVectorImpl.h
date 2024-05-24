// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorImpl.h                                 (C) 2000-2024 */
/*                                                                           */
/* Implémentation de 'IConstituentItemVectorImpl'.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_CONSTITUENTITEMVECTORIMPL_H
#define ARCANE_MATERIALS_INTERNAL_CONSTITUENTITEMVECTORIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/materials/ComponentItemVector.h"
#include "arcane/core/materials/internal/ConstituentItemLocalIdList.h"

#include "arcane/materials/internal/MeshComponentPartData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de ComponentItemVector.
 */
class ConstituentItemVectorImpl
: public IConstituentItemVectorImpl
, public ::Arccore::ReferenceCounterImpl
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  explicit ConstituentItemVectorImpl(IMeshComponent* component);
  explicit ConstituentItemVectorImpl(const ComponentItemVectorView& rhs);

 public:

  ConstituentItemVectorImpl(const ConstituentItemVectorImpl& rhs) = delete;
  ConstituentItemVectorImpl(ConstituentItemVectorImpl&& rhs) = delete;
  ConstituentItemVectorImpl& operator=(const ConstituentItemVectorImpl& rhs) = delete;

 private:

  ComponentItemVectorView _view() const override;
  ComponentPurePartItemVectorView _pureItems() const override
  {
    return m_part_data->pureView();
  }

  ComponentImpurePartItemVectorView _impureItems() const override
  {
    return m_part_data->impureView();
  }
  ConstArrayView<Int32> _localIds() const override { return m_items_local_id.constView(); }
  IMeshMaterialMng* _materialMng() const override { return m_material_mng; }
  IMeshComponent* _component() const override { return m_component; }
  ConstituentItemLocalIdListView _constituentItemListView() const override
  {
    return m_constituent_list->view();
  }
  ConstArrayView<MatVarIndex> _matvarIndexes() const override
  {
    return m_matvar_indexes;
  }
  void _setItems(ConstArrayView<ConstituentItemIndex> globals,
                 ConstArrayView<ConstituentItemIndex> multiples) override;

 public:

  IMeshMaterialMng* m_material_mng = nullptr;
  IMeshComponent* m_component = nullptr;
  UniqueArray<MatVarIndex> m_matvar_indexes;
  UniqueArray<Int32> m_items_local_id;
  std::unique_ptr<MeshComponentPartData> m_part_data;
  std::unique_ptr<ConstituentItemLocalIdList> m_constituent_list;
  ComponentItemSharedInfo* m_component_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

