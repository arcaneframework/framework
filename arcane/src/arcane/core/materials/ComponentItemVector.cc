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
/*!
 * \brief Implémentation de ComponentItemVector.
 */
class ConstituentItemVectorImpl
: public IConstituentItemVectorImpl
, public ReferenceCounterImpl
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  explicit ConstituentItemVectorImpl(IMeshComponent* component);
  ConstituentItemVectorImpl(IMeshComponent* component, const ConstituentItemLocalIdListView& constituent_list_view,
                            ConstArrayView<MatVarIndex> matvar_indexes, ConstArrayView<Int32> items_local_id);

 public:

  ConstituentItemVectorImpl(const ConstituentItemVectorImpl& rhs) = delete;
  ConstituentItemVectorImpl(ConstituentItemVectorImpl&& rhs) = delete;
  ConstituentItemVectorImpl& operator=(const ConstituentItemVectorImpl& rhs) = delete;

 private:

  void _setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                         ConstArrayView<MatVarIndex> multiples) override;
  void _setLocalIds(ConstArrayView<Int32> globals,
                    ConstArrayView<Int32> multiples) override;
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
                 ConstArrayView<ConstituentItemIndex> multiples) override
  {
    m_constituent_list->copyPureAndPartial(globals, multiples);
  }

 public:

  IMeshMaterialMng* m_material_mng = nullptr;
  IMeshComponent* m_component = nullptr;
  UniqueArray<MatVarIndex> m_matvar_indexes;
  UniqueArray<Int32> m_items_local_id;
  std::unique_ptr<MeshComponentPartData> m_part_data;
  std::unique_ptr<ConstituentItemLocalIdList> m_constituent_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(IMeshComponent* component)
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

ConstituentItemVectorImpl::
ConstituentItemVectorImpl(IMeshComponent* component, const ConstituentItemLocalIdListView& constituent_list_view,
                          ConstArrayView<MatVarIndex> matvar_indexes, ConstArrayView<Int32> items_local_id)
: ConstituentItemVectorImpl(component)
{
  m_constituent_list->copy(constituent_list_view);
  m_matvar_indexes.copy(matvar_indexes);
  m_items_local_id.copy(items_local_id);
  m_part_data->_setFromMatVarIndexes(matvar_indexes);
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

  m_part_data->_notifyValueIndexesChanged();
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(IMeshComponent* component)
: m_p(makeRef<IConstituentItemVectorImpl>(new ConstituentItemVectorImpl(component)))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVector::
ComponentItemVector(ComponentItemVectorView rhs)
: m_p(makeRef<IConstituentItemVectorImpl>(new ConstituentItemVectorImpl(rhs.component(), rhs._constituentItemListView(),
                                                                        rhs._matvarIndexes(), rhs._internalLocalIds())))
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
  m_p->_setItems(globals, multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setMatVarIndexes(ConstArrayView<MatVarIndex> globals,
                  ConstArrayView<MatVarIndex> multiples)
{
  m_p->_setMatVarIndexes(globals, multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemVector::
_setLocalIds(ConstArrayView<Int32> globals, ConstArrayView<Int32> multiples)
{
  m_p->_setLocalIds(globals, multiples);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ComponentItemVector::
view() const
{
  return m_p->_view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentPurePartItemVectorView ComponentItemVector::
pureItems() const
{
  return m_p->_pureItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentImpurePartItemVectorView ComponentItemVector::
impureItems() const
{
  return m_p->_impureItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemLocalIdListView ComponentItemVector::
_constituentItemListView() const
{
  return m_p->_constituentItemListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> ComponentItemVector::
_matvarIndexes() const
{
  return m_p->_matvarIndexes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> ComponentItemVector::
_localIds() const
{
  return m_p->_localIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialMng* ComponentItemVector::
_materialMng() const
{
  return m_p->_materialMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshComponent* ComponentItemVector::
_component() const
{
  return m_p->_component();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
