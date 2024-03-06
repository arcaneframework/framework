﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternal.h                                     (C) 2000-2024 */
/*                                                                           */
/* Partie interne d'une maille multi-matériau.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_COMPONENTITEMINTERNAL_H
#define ARCANE_CORE_MATERIALS_COMPONENTITEMINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternal.h"
#include "arcane/core/Item.h"
#include "arcane/core/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshEnvironment;
class MeshComponentData;
class AllEnvData;
class MeshMaterialMng;
class ComponentItemInternalData;

namespace matimpl
{
  class ConstituentItemBase;
}

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
  friend class ComponentItemSharedInfo;

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
class ARCANE_CORE_EXPORT ComponentItemSharedInfo
: private ComponentItemSharedInfoStorageView
{
  friend class ComponentItemInternalData;
  friend class CellComponentCellEnumerator;
  friend class ConstituentItemLocalIdList;
  friend class ConstituentItemLocalIdListView;
  friend matimpl::ConstituentItemBase;
  friend class ComponentCell;
  friend class CellToAllEnvCellConverter;
  friend class AllEnvCellVectorView;

  static const int MAT_INDEX_OFFSET = 10;

 private:

  //! Pour l'entité nulle
  static ComponentItemSharedInfo null_shared_info;
  static ComponentItemSharedInfo* null_shared_info_pointer;
  static ComponentItemSharedInfo* _nullInstance() { return null_shared_info_pointer; }

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
  inline void _setNbSubConstituent(ConstituentItemIndex id, Int16 n)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_nb_sub_constituent_item_data[id.localId()] = n;
  }
  inline ARCCORE_HOST_DEVICE Int16 _componentId(ConstituentItemIndex id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_component_id_data[id.localId()];
  }
  inline void _setComponentId(ConstituentItemIndex id, Int16 component_id)
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
  void _setGlobalItem(ConstituentItemIndex id, ItemLocalId global_item_lid)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_global_item_local_id_data[id.localId()] = global_item_lid.localId();
  }
  matimpl::ConstituentItemBase _superItemBase(ConstituentItemIndex id) const;

  ARCCORE_HOST_DEVICE void _setSuperItem(ConstituentItemIndex id, ConstituentItemIndex super_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_super_component_item_local_id_data[id.localId()] = super_id;
  }
  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(ConstituentItemIndex id,Int32 sub_index) const;

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

}

namespace Arcane::Materials::matimpl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations générique sur une entité d'un constituant.
 *
 * Cette classe est le pendant de ItemInternal pour la gestion des matériaux
 * et des milieux. Elle ne doit en principe pas être utilisée directement, sauf
 * par les classes de Arcane. Il vaut mieux utiliser les
 * classes ComponentCell,  MatCell, EnvCell ou AllEnvCell.
 */
class ARCANE_CORE_EXPORT ConstituentItemBase
{
  friend Arcane::Materials::ComponentCell;
  friend Arcane::Materials::AllEnvCell;
  friend Arcane::Materials::EnvCell;
  friend Arcane::Materials::MatCell;
  friend Arcane::Materials::AllEnvData;
  friend Arcane::Materials::MeshMaterialMng;
  friend Arcane::Materials::ComponentItemSharedInfo;

  friend Arcane::Materials::MeshEnvironment;
  friend Arcane::Materials::MeshComponentData;
  friend Arcane::Materials::ComponentItemInternalData;

 private:

  ARCCORE_HOST_DEVICE constexpr ConstituentItemBase(ComponentItemSharedInfo* shared_info, ConstituentItemIndex id)
  : m_constituent_item_index(id)
  , m_shared_info(shared_info)
  {
  }

 public:

  //! Indexeur dans les variables matériaux
  ARCCORE_HOST_DEVICE MatVarIndex variableIndex() const
  {
    return m_shared_info->_varIndex(m_constituent_item_index);
  }

  ARCCORE_HOST_DEVICE ConstituentItemIndex constituentItemIndex() const
  {
    return m_constituent_item_index;
  }

  //! Identifiant du composant
  ARCCORE_HOST_DEVICE Int32 componentId() const
  {
    return m_shared_info->_componentId(m_constituent_item_index);
  }

  //! Indique s'il s'agit de la maille nulle.
  inline ARCCORE_HOST_DEVICE constexpr bool null() const
  {
    return m_constituent_item_index.isNull();
  }

  /*!
   * \brief Composant associé.
   *
   * Cet appel n'est valide que pour les mailles matériaux ou milieux. Si on souhaite
   * un appel valide pour toutes les 'ComponentItem', il faut utiliser componentId().
   */
  inline IMeshComponent* component() const
  {
    return m_shared_info->_component(m_constituent_item_index);
  }

  //! Nombre de sous-composants.
  ARCCORE_HOST_DEVICE Int32 nbSubItem() const
  {
    return m_shared_info->_nbSubConstituent(m_constituent_item_index);
  }


  //! Entité globale correspondante.
  inline impl::ItemBase globalItemBase() const
  {
    return m_shared_info->_globalItemBase(m_constituent_item_index);
  }

  inline ARCCORE_HOST_DEVICE constexpr Int32 level() const
  {
    return m_shared_info->m_level;
  }

  //! Numéro unique de l'entité component
  inline Int64 componentUniqueId() const
  {
    return m_shared_info->_componentUniqueId(m_constituent_item_index);
  }

 public:

  ARCCORE_HOST_DEVICE constexpr friend bool
  operator==(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return ((a.m_constituent_item_index==b.m_constituent_item_index) && (a.m_shared_info==b.m_shared_info));
  }
  ARCCORE_HOST_DEVICE constexpr friend bool
  operator!=(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return !(a==b);
  }

 private:

  //! Positionne l'indexeur dans les variables matériaux.
  inline void _setVariableIndex(MatVarIndex index)
{
  m_shared_info->_setVarIndex(m_constituent_item_index, index);
}

  //! Composant supérieur (0 si aucun)
  inline matimpl::ConstituentItemBase _superItemBase() const;

  inline void _setSuperAndGlobalItem(ConstituentItemIndex cii, ItemLocalId ii)
  {
    m_shared_info->_setSuperItem(m_constituent_item_index, cii);
    m_shared_info->_setGlobalItem(m_constituent_item_index, ii);
  }

  inline void _setGlobalItem(ItemLocalId ii)
  {
    m_shared_info->_setGlobalItem(m_constituent_item_index, ii);
  }

  //! Première entité sous-composant.
  inline ARCCORE_HOST_DEVICE ConstituentItemIndex _firstSubItemLocalId() const
  {
    return m_shared_info->_firstSubConstituentLocalId(m_constituent_item_index);
  }

  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 i) const;

  //! Positionne le nombre de sous-composants.
  void _setNbSubItem(Int16 nb_sub_item)
  {
    m_shared_info->_setNbSubConstituent(m_constituent_item_index, nb_sub_item);
  }

  //! Positionne le premier sous-composant.
  void _setFirstSubItem(ConstituentItemIndex first_sub_item)
  {
    m_shared_info->_setFirstSubConstituentLocalId(m_constituent_item_index, first_sub_item);
  }

  void _setComponent(Int16 component_id)
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
  return matimpl::ConstituentItemBase(this,id);
}

inline matimpl::ConstituentItemBase ComponentItemSharedInfo::
_superItemBase(ConstituentItemIndex id) const
{
  ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
  ConstituentItemIndex super_local_id(m_super_component_item_local_id_data[id.localId()]);
  return m_super_component_item_shared_info->_item(super_local_id);
}

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase ComponentItemSharedInfo::
_subItemBase(ConstituentItemIndex id,Int32 sub_index) const
{
  ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
  ConstituentItemIndex lid(m_first_sub_constituent_item_id_data[id.localId()].localId() + sub_index);
  return m_sub_component_item_shared_info->_item(lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline matimpl::ConstituentItemBase matimpl::ConstituentItemBase::
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
 * \brief Vue sur une instance de ConstituentItemLocalIdList.
 *
 * Les instances de ces classes sont notamment utilisées pour les énumérateurs
 * sur les constituants.
 */
class ARCANE_CORE_EXPORT ConstituentItemLocalIdListView
{
  friend class ConstituentItemLocalIdList;
  friend class ComponentItemVectorView;
  friend class MeshComponentPartData;
  friend class ComponentPartItemVectorView;
  friend class ComponentPartCellEnumerator;
  friend class ComponentCellEnumerator;

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

  matimpl::ConstituentItemBase _constituenItemBase(Int32 index) const
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
  //! Pour les tests, vérifie que les vues pointent vers les mêmes données
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

  // NOTE: Cette classe est wrappé directement en C#.
  // Si on modifie les champs de cette classe il faut modifier le type correspondant
  // dans le wrappeur.
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
