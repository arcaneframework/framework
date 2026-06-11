// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemBase.h                                       (C) 2000-2024 */
/*                                                                           */
/* General information about a constituent entity.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEMBASE_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEMBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternal.h"
#include "arcane/core/Item.h"
#include "arcane/core/materials/ConstituentItemLocalId.h"
#include "arcane/core/materials/ComponentItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials::matimpl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief General information about a constituent entity.
 *
 * This class is the counterpart of ItemInternal for managing materials
 * and environments. In principle, it should not be used directly, except
 * by Arcane classes. It is better to use the
 * ComponentCell, MatCell, EnvCell, or AllEnvCell classes.
 */
class ARCANE_CORE_EXPORT ConstituentItemBase
{
  friend ComponentCell;
  friend AllEnvCell;
  friend EnvCell;
  friend MatCell;
  friend AllEnvData;
  friend MeshMaterialMng;
  friend ComponentItemSharedInfo;

  friend MeshEnvironment;
  friend MeshComponentData;
  friend ComponentItemInternalData;

 private:

  ARCCORE_HOST_DEVICE constexpr ConstituentItemBase(ComponentItemSharedInfo* shared_info, ConstituentItemIndex id)
  : m_constituent_item_index(id)
  , m_shared_info(shared_info)
  {
  }

 public:

  //! Indexer in material variables
  ARCCORE_HOST_DEVICE MatVarIndex variableIndex() const
  {
    return m_shared_info->_varIndex(m_constituent_item_index);
  }

  ARCCORE_HOST_DEVICE ConstituentItemIndex constituentItemIndex() const
  {
    return m_constituent_item_index;
  }

  //! Component identifier
  ARCCORE_HOST_DEVICE Int32 componentId() const
  {
    return m_shared_info->_componentId(m_constituent_item_index);
  }

  //! Indicates if it is the null cell.
  inline ARCCORE_HOST_DEVICE constexpr bool null() const
  {
    return m_constituent_item_index.isNull();
  }

  /*!
   * \brief Associated component.
   *
   * This call is only valid for material or environment cells. If you want
   * a valid call for all 'ComponentItem's, you must use componentId().
   */
  inline IMeshComponent* component() const
  {
    return m_shared_info->_component(m_constituent_item_index);
  }

  //! Number of sub-components.
  ARCCORE_HOST_DEVICE Int32 nbSubItem() const
  {
    return m_shared_info->_nbSubConstituent(m_constituent_item_index);
  }

  //! Corresponding global entity.
  inline impl::ItemBase globalItemBase() const
  {
    return m_shared_info->_globalItemBase(m_constituent_item_index);
  }

  inline ARCCORE_HOST_DEVICE constexpr Int32 level() const
  {
    return m_shared_info->m_level;
  }

  //! Unique ID of the component entity
  inline Int64 componentUniqueId() const
  {
    return m_shared_info->_componentUniqueId(m_constituent_item_index);
  }

 public:

  ARCCORE_HOST_DEVICE constexpr friend bool
  operator==(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return ((a.m_constituent_item_index == b.m_constituent_item_index) && (a.m_shared_info == b.m_shared_info));
  }
  ARCCORE_HOST_DEVICE constexpr friend bool
  operator!=(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return !(a == b);
  }

 private:

  //! Positions the indexer in material variables.
  ARCCORE_HOST_DEVICE inline void _setVariableIndex(MatVarIndex index)
  {
    m_shared_info->_setVarIndex(m_constituent_item_index, index);
  }

  //! Parent component (0 if none)
  ARCCORE_HOST_DEVICE inline matimpl::ConstituentItemBase _superItemBase() const;

  ARCCORE_HOST_DEVICE inline void _setSuperAndGlobalItem(ConstituentItemIndex cii, ItemLocalId ii)
  {
    m_shared_info->_setSuperItem(m_constituent_item_index, cii);
    m_shared_info->_setGlobalItem(m_constituent_item_index, ii);
  }

  ARCCORE_HOST_DEVICE inline void _setGlobalItem(ItemLocalId ii)
  {
    m_shared_info->_setGlobalItem(m_constituent_item_index, ii);
  }

  //! First sub-component entity.
  inline ARCCORE_HOST_DEVICE ConstituentItemIndex _firstSubItemLocalId() const
  {
    return m_shared_info->_firstSubConstituentLocalId(m_constituent_item_index);
  }

  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 i) const;

  //! Positions the number of sub-components.
  ARCCORE_HOST_DEVICE void _setNbSubItem(Int16 nb_sub_item)
  {
    m_shared_info->_setNbSubConstituent(m_constituent_item_index, nb_sub_item);
  }

  //! Positions the first sub-component.
  ARCCORE_HOST_DEVICE void _setFirstSubItem(ConstituentItemIndex first_sub_item)
  {
    m_shared_info->_setFirstSubConstituentLocalId(m_constituent_item_index, first_sub_item);
  }

  ARCCORE_HOST_DEVICE void _setComponent(Int16 component_id)
  {
    m_shared_info->_setComponentId(m_constituent_item_index, component_id);
  }

 private:

  ConstituentItemIndex m_constituent_item_index;
  ComponentItemSharedInfo* m_shared_info = ComponentItemSharedInfo::null_shared_info_pointer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials::matimpl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr matimpl::ConstituentItemBase ComponentItemSharedInfo::
_item(ConstituentItemIndex id)
{
  return matimpl::ConstituentItemBase(this, id);
}

ARCCORE_HOST_DEVICE inline matimpl::ConstituentItemBase ComponentItemSharedInfo::
_superItemBase(ConstituentItemIndex id) const
{
  ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
  ConstituentItemIndex super_local_id(m_super_component_item_local_id_data[id.localId()]);
  return m_super_component_item_shared_info->_item(super_local_id);
}

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase ComponentItemSharedInfo::
_subItemBase(ConstituentItemIndex id, Int32 sub_index) const
{
  ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
  ConstituentItemIndex lid(m_first_sub_constituent_item_id_data[id.localId()].localId() + sub_index);
  return m_sub_component_item_shared_info->_item(lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase matimpl::ConstituentItemBase::
_superItemBase() const
{
  return m_shared_info->_superItemBase(m_constituent_item_index);
}

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase matimpl::ConstituentItemBase::
_subItemBase(Int32 i) const
{
  return m_shared_info->_subItemBase(m_constituent_item_index, i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief View of a ConstituentItemLocalIdList instance.
 *
 * Instances of these classes are notably used for enumerators
 * over constituents.
 */
class ARCANE_CORE_EXPORT ConstituentItemLocalIdListView
{
  friend class ConstituentItemLocalIdList;
  friend class ComponentItemVectorView;
  friend class MeshComponentPartData;
  friend class ComponentPartItemVectorView;
  friend class ComponentPartCellEnumerator;
  friend class ComponentCellEnumerator;
  friend class MeshEnvironment;

 private:

  ConstituentItemLocalIdListView() = default;
  ConstituentItemLocalIdListView(ComponentItemSharedInfo* shared_info,
                                 ConstArrayView<ConstituentItemIndex> ids)
  : m_component_shared_info(shared_info)
  , m_ids(ids)
  {
#ifdef ARCANE_CHECK
    _checkCoherency();
#endif
  }

 private:

  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _constituenItemBase(Int32 index) const
  {
    return m_component_shared_info->_item(m_ids[index]);
  }
  MatVarIndex _matVarIndex(Int32 index) const
  {
    return m_component_shared_info->_varIndex(m_ids[index]);
  }
  ConstituentItemLocalIdListView _subView(Int32 begin, Int32 size) const
  {
    return { m_component_shared_info, m_ids.subView(begin, size) };
  }
  //! For tests, verifies that the views point to the same data
  bool _isSamePointerData(const ConstituentItemLocalIdListView& rhs) const
  {
    return (m_ids.data() == rhs.m_ids.data());
  }
  friend bool operator==(const ConstituentItemLocalIdListView& a,
                         const ConstituentItemLocalIdListView& b)
  {
    bool t1 = a.m_component_shared_info == b.m_component_shared_info;
    bool t2 = a.m_ids == b.m_ids;
    return (t1 && t2);
  }
  friend bool operator!=(const ConstituentItemLocalIdListView& a,
                         const ConstituentItemLocalIdListView& b)
  {
    return (!(a == b));
  }

 private:

  // NOTE: This class is wrapped directly in C#.
  // If the fields of this class are modified, the corresponding type must be modified
  // in the wrapper.
  ComponentItemSharedInfo* m_component_shared_info = nullptr;
  ConstArrayView<ConstituentItemIndex> m_ids;

 private:

  void _checkCoherency() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
