// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemSharedInfo.h                                 (C) 2000-2024 */
/*                                                                           */
/* Informations partagées pour les structures de 'ConstituentItem'           */
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
 * \brief Index d'une entité constituant dans la liste des entités constituants.
 *
 * L'index est propre à chaque type d'entité consituant (AllEnvCell, EnvCell, MatCell).
 * La liste est gérée par ComponentIemtInternalData.
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
 * \brief Conteneur pour les données des constituants.
 */
class ARCANE_CORE_EXPORT ComponentItemSharedInfoStorageView
{
  // Les champs de cette classe sont des tableaux dont la taille est
  // \a m_storage_size et qui peuvent être indexés par une entité nulle
  // (ConstituentItemIndex==(-1)).
  // Le conteneur est géré par ComponenItemInternalData.
  // Seuls ComponentItemSharedInfo et ComponenItemInternalData
  // doivent accéder aux champs de cette classe

  // TODO: Utiliser stockage avec un seul élément pour le nullComponent

  friend class ComponentItemInternalData;
  friend ConstituentItemSharedInfo;

 private:

  Int32 m_storage_size = 0;
  //! Id de la première entité sous-constituant
  ConstituentItemIndex* m_first_sub_constituent_item_id_data = nullptr;
  //! Index du constituant (IMeshComponent)
  Int16* m_component_id_data = nullptr;
  //! Nombre d'entités sous-constituant
  Int16* m_nb_sub_constituent_item_data = nullptr;
  //! localId() de l'entité globale associée
  Int32* m_global_item_local_id_data = nullptr;
  //! Id de l'entité sous-constituant parente
  ConstituentItemIndex* m_super_component_item_local_id_data = nullptr;
  //! MatVarIndex de l'entité
  MatVarIndex* m_var_index_data = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations partagées sur les 'ComponentItem'.
 *
 * Il y a 3 instances de cette classe : une pour les AllEnvCell, une pour les
 * EnvCell et une pour les MatCell. Ces instances sont gérées par la classe
 * ComponentItemInternalData. Il est possible de conserver un pointeur sur
 * les intances de cette classe car ils sont valides durant toute la vie
 * d'un MeshMaterialMng.
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

  //! Pour l'entité nulle
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

  //! Numéro unique de l'entité component
  Int64 _componentUniqueId(ConstituentItemIndex id) const
  {
    // TODO: Vérifier que arrayIndex() ne dépasse pas (1<<MAT_INDEX_OFFSET)
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

  // NOTE : Cette classe est partagée avec le wrapper C#
  // Toute modification de la structure interne doit être reportée
  // dans la structure C# correspondante
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
