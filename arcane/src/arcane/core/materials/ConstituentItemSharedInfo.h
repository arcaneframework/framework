// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemSharedInfo.h                                 (C) 2000-2024 */
/*                                                                           */
/* Shared information for 'ConstituentItem' structures                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEMSHAREDINFO_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEMSHAREDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternal.h"
#include "arcane/core/Item.h"
#include "arcane/core/materials/ConstituentItemLocalId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Index of a constituent entity in the list of constituent entities.
 *
 * The index is specific to each type of constituent entity (AllEnvCell, EnvCell, MatCell).
 * The list is managed by ComponentIemtInternalData.
 */
class ARCANE_CORE_EXPORT ConstituentItemIndex
{
 public:

  ConstituentItemIndex() = default;
  explicit ARCCORE_HOST_DEVICE constexpr ConstituentItemIndex(Int32 id)
  : m_id(id)
  {}
  ARCCORE_HOST_DEVICE constexpr Int32 localId() const { return m_id; }
  ARCCORE_HOST_DEVICE friend constexpr bool operator==(ConstituentItemIndex a,
                                                       ConstituentItemIndex b)
  {
    return a.m_id == b.m_id;
  }
  ARCCORE_HOST_DEVICE friend constexpr bool operator!=(ConstituentItemIndex a,
                                                       ConstituentItemIndex b)
  {
    return a.m_id != b.m_id;
  }
  ARCANE_CORE_EXPORT friend std::ostream&
  operator<<(std::ostream& o, const ConstituentItemIndex& id);

  ARCCORE_HOST_DEVICE constexpr bool isNull() const { return m_id == (-1); }

 private:

  Int32 m_id = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Container for constituent data.
 */
class ARCANE_CORE_EXPORT ComponentItemSharedInfoStorageView
{
  // The fields of this class are arrays whose size is
  // \a m_storage_size and which can be indexed by a null entity
  // (ConstituentItemIndex==(-1)).
  // The container is managed by ComponenItemInternalData.
  // Only ComponentItemSharedInfo and ComponenItemInternalData
  // should access the fields of this class

  // TODO: Use single-element storage for the nullComponent

  friend class ComponentItemInternalData;
  friend ConstituentItemSharedInfo;

 private:

  Int32 m_storage_size = 0;
  //! Id of the first sub-constituent entity
  ConstituentItemIndex* m_first_sub_constituent_item_id_data = nullptr;
  //! Index of the constituent (IMeshComponent)
  Int16* m_component_id_data = nullptr;
  //! Number of sub-constituent entities
  Int16* m_nb_sub_constituent_item_data = nullptr;
  //! localId() of the associated global entity
  Int32* m_global_item_local_id_data = nullptr;
  //! Id of the parent sub-constituent entity
  ConstituentItemIndex* m_super_component_item_local_id_data = nullptr;
  //! MatVarIndex of the entity
  MatVarIndex* m_var_index_data = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Shared information about 'ComponentItem'.
 *
 * There are 3 instances of this class: one for AllEnvCell, one for
 * EnvCell, and one for MatCell. These instances are managed by the class
 * ComponentItemInternalData. It is possible to keep a pointer to
 * the instances of this class because they are valid throughout the life
 * of a MeshMaterialMng.
 */
class ARCANE_CORE_EXPORT ConstituentItemSharedInfo
: private ComponentItemSharedInfoStorageView
{
  friend ComponentItemInternalData;
  friend CellComponentCellEnumerator;
  friend ConstituentItemLocalIdList;
  friend ConstituentItemLocalIdListView;
  friend matimpl::ConstituentItemBase;
  friend ConstituentItem;
  friend CellToAllEnvCellConverter;
  friend AllEnvCellVectorView;
  friend ConstituentItemVectorImpl;

  static const int MAT_INDEX_OFFSET = 10;

 private:

  //! For the null entity
  static ComponentItemSharedInfo null_shared_info;
  static ComponentItemSharedInfo* null_shared_info_pointer;
  static ComponentItemSharedInfo* _nullInstance() { return null_shared_info_pointer; }
  static void _setNullInstance();

 private:

  inline constexpr matimpl::ConstituentItemBase _item(ConstituentItemIndex id);
  inline ARCCORE_HOST_DEVICE ConstituentItemIndex _firstSubConstituentLocalId(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_first_sub_constituent_item_id_data[id.localId()];
  }
  inline ARCCORE_HOST_DEVICE void
  _setFirstSubConstituentLocalId(ConstituentItemIndex id, ConstituentItemIndex first_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_first_sub_constituent_item_id_data[id.localId()] = first_id;
  }
  inline ARCCORE_HOST_DEVICE Int16 _nbSubConstituent(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_nb_sub_constituent_item_data[id.localId()];
  }
  ARCCORE_HOST_DEVICE inline void _setNbSubConstituent(ConstituentItemIndex id, Int16 n)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_nb_sub_constituent_item_data[id.localId()] = n;
  }
  inline ARCCORE_HOST_DEVICE Int16 _componentId(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_component_id_data[id.localId()];
  }
  ARCCORE_HOST_DEVICE inline void _setComponentId(ConstituentItemIndex id, Int16 component_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_component_id_data[id.localId()] = component_id;
  }
  IMeshComponent* _component(ConstituentItemIndex id) const
  {
    return m_components[_componentId(id)];
  }
  impl::ItemBase _globalItemBase(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return impl::ItemBase(m_global_item_local_id_data[id.localId()], m_item_shared_info);
  }
  ARCCORE_HOST_DEVICE Int32 _globalItemId(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_global_item_local_id_data[id.localId()];
  }
  ARCCORE_HOST_DEVICE void _setGlobalItem(ConstituentItemIndex id, ItemLocalId global_item_lid)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_global_item_local_id_data[id.localId()] = global_item_lid.localId();
  }
  ARCCORE_HOST_DEVICE inline matimpl::ConstituentItemBase _superItemBase(ConstituentItemIndex id) const;

  ARCCORE_HOST_DEVICE void _setSuperItem(ConstituentItemIndex id, ConstituentItemIndex super_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_super_component_item_local_id_data[id.localId()] = super_id;
  }
  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(ConstituentItemIndex id, Int32 sub_index) const;

  ARCCORE_HOST_DEVICE MatVarIndex _varIndex(ConstituentItemIndex id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_var_index_data[id.localId()];
  }
  ARCCORE_HOST_DEVICE void _setVarIndex(ConstituentItemIndex id, MatVarIndex mv_index)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_var_index_data[id.localId()] = mv_index;
  }

  //! Unique ID number of the component entity
  Int64 _componentUniqueId(ConstituentItemIndex id) const
  {
    // TODO: Check that arrayIndex() does not exceed (1<<MAT_INDEX_OFFSET)
    impl::ItemBase item_base(_globalItemBase(id));
    return (Int64)m_var_index_data[id.localId()].arrayIndex() + ((Int64)item_base.uniqueId() << MAT_INDEX_OFFSET);
  }

  ARCCORE_HOST_DEVICE void _reset(ConstituentItemIndex id)
  {
    Int32 local_id = id.localId();
    ARCCORE_CHECK_RANGE(local_id, -1, m_storage_size);

    m_var_index_data[local_id].reset();
    m_first_sub_constituent_item_id_data[local_id] = {};
    m_nb_sub_constituent_item_data[local_id] = 0;
    m_component_id_data[local_id] = -1;
    m_global_item_local_id_data[local_id] = NULL_ITEM_LOCAL_ID;
    m_super_component_item_local_id_data[local_id] = {};
  }

 private:

  // NOTE: This class is shared with the C# wrapper
  // Any modification to the internal structure must be reported
  // in the corresponding C# structure
  ItemSharedInfo* m_item_shared_info = ItemSharedInfo::nullInstance();
  Int16 m_level = (-1);
  ConstArrayView<IMeshComponent*> m_components;
  ComponentItemSharedInfo* m_super_component_item_shared_info = null_shared_info_pointer;
  ComponentItemSharedInfo* m_sub_component_item_shared_info = null_shared_info_pointer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ConstituentItemBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
