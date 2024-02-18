// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
namespace matimpl
{
  class ConstituentItemBase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCANE_CORE_EXPORT ComponentItemInternalLocalId
{
 public:

  ComponentItemInternalLocalId() = default;
  explicit ARCCORE_HOST_DEVICE constexpr ComponentItemInternalLocalId(Int32 id)
  : m_id(id)
  {}
  ARCCORE_HOST_DEVICE constexpr Int32 localId() const { return m_id; }
  ARCCORE_HOST_DEVICE friend constexpr bool operator==(ComponentItemInternalLocalId a,
                                                       ComponentItemInternalLocalId b)
  {
    return a.m_id == b.m_id;
  }
  ARCCORE_HOST_DEVICE friend constexpr bool operator!=(ComponentItemInternalLocalId a,
                                                       ComponentItemInternalLocalId b)
  {
    return a.m_id != b.m_id;
  }
  ARCANE_CORE_EXPORT friend std::ostream&
  operator<<(std::ostream& o, const ComponentItemInternalLocalId& id);

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
  // (ComponentItemInternalLocalId==(-1)).
  // Le conteneur est géré par ComponenItemInternalData.
  // Seuls ComponentItemSharedInfo et ComponenItemInternalData
  // doivent accéder aux champs de cette classe

  // TODO: Utiliser stockage avec un seul élément pour le nullComponent

  friend class ComponentItemInternalData;
  friend class ComponentItemSharedInfo;

 private:

  Int32 m_storage_size = 0;
  //! Id de la première entité sous-constituant
  ComponentItemInternalLocalId* m_first_sub_constituent_item_id_data = nullptr;
  //! Index du constituant (IMeshComponent)
  Int16* m_component_id_data = nullptr;
  //! Nombre d'entités sous-constituant
  Int16* m_nb_sub_constituent_item_data = nullptr;
  //! localId() de l'entité globale associée
  Int32* m_global_item_local_id_data = nullptr;
  //! Id de l'entité sous-constituant parente
  ComponentItemInternalLocalId* m_super_component_item_local_id_data = nullptr;
  //! MatVarIndex de l'entité
  MatVarIndex* m_var_index_data = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations partagées sur les 'ComponentItemInternal'.
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
  friend class ComponentItemInternal;
  friend class ComponentItemInternalData;
  friend class CellComponentCellEnumerator;
  friend class ConstituentItemLocalIdList;
  friend class ConstituentItemLocalIdListView;
  friend class matimpl::ConstituentItemBase;

  static const int MAT_INDEX_OFFSET = 10;

 private:

  //! Pour l'entité nulle
  static ComponentItemSharedInfo null_shared_info;
  static ComponentItemSharedInfo* null_shared_info_pointer;
  static ComponentItemSharedInfo* _nullInstance() { return null_shared_info_pointer; }

 private:

  inline constexpr ComponentItemInternal* _itemInternal(ComponentItemInternalLocalId id);
  inline constexpr matimpl::ConstituentItemBase _item(ComponentItemInternalLocalId id);
  inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _firstSubConstituentLocalId(ComponentItemInternalLocalId id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_first_sub_constituent_item_id_data[id.localId()];
  }
  inline ARCCORE_HOST_DEVICE void
  _setFirstSubConstituentLocalId(ComponentItemInternalLocalId id, ComponentItemInternalLocalId first_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_first_sub_constituent_item_id_data[id.localId()] = first_id;
  }
  inline ARCCORE_HOST_DEVICE Int16 _nbSubConstituent(ComponentItemInternalLocalId id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_nb_sub_constituent_item_data[id.localId()];
  }
  inline void _setNbSubConstituent(ComponentItemInternalLocalId id, Int16 n)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_nb_sub_constituent_item_data[id.localId()] = n;
  }
  inline ARCCORE_HOST_DEVICE Int16 _componentId(ComponentItemInternalLocalId id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_component_id_data[id.localId()];
  }
  inline void _setComponentId(ComponentItemInternalLocalId id, Int16 component_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_component_id_data[id.localId()] = component_id;
  }
  IMeshComponent* _component(ComponentItemInternalLocalId id) const
  {
    return m_components[_componentId(id)];
  }
  impl::ItemBase _globalItemBase(ComponentItemInternalLocalId id) const
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return impl::ItemBase(m_global_item_local_id_data[id.localId()], m_item_shared_info);
  }
  void _setGlobalItem(ComponentItemInternalLocalId id, ItemLocalId global_item_lid)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_global_item_local_id_data[id.localId()] = global_item_lid.localId();
  }
  matimpl::ConstituentItemBase _superItemBase(ComponentItemInternalLocalId id) const;

  ARCCORE_HOST_DEVICE void _setSuperItem(ComponentItemInternalLocalId id, ComponentItemInternalLocalId super_id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_super_component_item_local_id_data[id.localId()] = super_id;
  }

  ARCCORE_HOST_DEVICE MatVarIndex _varIndex(ComponentItemInternalLocalId id)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    return m_var_index_data[id.localId()];
  }
  ARCCORE_HOST_DEVICE void _setVarIndex(ComponentItemInternalLocalId id, MatVarIndex mv_index)
  {
    ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
    m_var_index_data[id.localId()] = mv_index;
  }

  //! Numéro unique de l'entité component
  Int64 _componentUniqueId(ComponentItemInternalLocalId id) const
  {
    // TODO: Vérifier que arrayIndex() ne dépasse pas (1<<MAT_INDEX_OFFSET)
    impl::ItemBase item_base(_globalItemBase(id));
    return (Int64)m_var_index_data[id.localId()].arrayIndex() + ((Int64)item_base.uniqueId() << MAT_INDEX_OFFSET);
  }

  inline void _reset(ComponentItemInternalLocalId id)
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
  ArrayView<ComponentItemInternal> m_component_item_internal_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

namespace Arcane::Materials::matimpl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations générique sur une entité d'un constituant.
 */
class ARCANE_CORE_EXPORT ConstituentItemBase
{
  friend Arcane::Materials::ComponentCell;
  friend Arcane::Materials::AllEnvCell;
  friend Arcane::Materials::EnvCell;
  friend Arcane::Materials::MatCell;
  friend Arcane::Materials::AllEnvData;
  friend Arcane::Materials::MeshMaterialMng;

  friend Arcane::Materials::MeshEnvironment;
  friend Arcane::Materials::MeshComponentData;

 public:

  ARCCORE_HOST_DEVICE constexpr ConstituentItemBase(ComponentItemInternal* component_item)
  : m_component_item(component_item)
  {
  }

 private:

  inline constexpr ConstituentItemBase(ComponentItemSharedInfo* shared_info, ComponentItemInternalLocalId id);

 public:

  //! Indexeur dans les variables matériaux
  inline ARCCORE_HOST_DEVICE MatVarIndex variableIndex() const;

  //! Identifiant du composant
  inline ARCCORE_HOST_DEVICE Int32 componentId() const;

  //! Indique s'il s'agit de la maille nulle.
  inline ARCCORE_HOST_DEVICE constexpr bool null() const;

  /*!
   * \brief Composant associé.
   *
   * Cet appel n'est valide que pour les mailles matériaux ou milieux. Si on souhaite
   * un appel valide pour toutes les 'ComponentItem', il faut utiliser componentId().
   */
  inline IMeshComponent* component() const;

  //! Nombre de sous-composants.
  inline ARCCORE_HOST_DEVICE Int32 nbSubItem() const;

  //! Entité globale correspondante.
  inline impl::ItemBase globalItemBase() const;

  inline ARCCORE_HOST_DEVICE constexpr Int32 level() const;

  //! Numéro unique de l'entité component
  inline Int64 componentUniqueId() const;

 public:

  ARCCORE_HOST_DEVICE constexpr friend bool
  operator==(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return a.m_component_item == b.m_component_item;
  }
  ARCCORE_HOST_DEVICE constexpr friend bool
  operator!=(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return a.m_component_item != b.m_component_item;
  }

 private:

  //! Positionne l'indexeur dans les variables matériaux.
  inline void _setVariableIndex(MatVarIndex index);

  //! Composant supérieur (0 si aucun)
  inline matimpl::ConstituentItemBase _superItemBase() const;

  inline void _setSuperAndGlobalItem(ComponentItemInternalLocalId cii, ItemLocalId ii);

  inline void _setGlobalItem(ItemLocalId ii);

  //! Première entité sous-composant.
  inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _firstSubItemLocalId() const;

  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 i) const;

  //! Positionne le nombre de sous-composants.
  inline void _setNbSubItem(Int16 nb_sub_item);

  //! Positionne le premier sous-composant.
  inline void _setFirstSubItem(ComponentItemInternalLocalId first_sub_item);

  inline void _setComponent(Int16 component_id);

  inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _internalLocalId() const;

  inline void _reset(ComponentItemInternalLocalId id, ComponentItemSharedInfo* shared_info);

 private:

  ARCCORE_HOST_DEVICE constexpr ComponentItemInternal* _internal() const { return m_component_item; }

 private:

  ComponentItemInternal* m_component_item = nullptr;

 private:

  inline ARCCORE_HOST_DEVICE ComponentItemSharedInfo* _sharedInfo() const;
  inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _localId() const;
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
/*!
 * \internal
 * \brief Partie interne d'une maille matériau ou milieu.
 *
 * Cette classe est le pendant de ItemInternal pour la gestion des matériaux
 * et des milieux. Elle ne doit en principe pas être utilisée directement, sauf
 * par les classes de Arcane. Il vaut mieux utiliser les
 * classes ComponentCell,  MatCell, EnvCell ou AllEnvCell.
 *
 * \todo pour économiser la mémoire, utiliser un ComponentItemSharedInfo
 * pour stocker une fois les infos multiples.
 */
class ARCANE_CORE_EXPORT ComponentItemInternal
{
  friend class ComponentCell;
  friend class MatCell;
  friend class EnvCell;
  friend class ComponentCell;
  friend class AllEnvCell;
  friend class CellComponentCellEnumerator;
  friend class ComponentItemInternalData;
  friend class MeshComponentData;
  friend class MeshEnvironment;
  friend class AllEnvData;
  friend class MeshMaterialMng;
  friend class ConstituentItemLocalIdList;
  friend class ConstituentItemLocalIdListView;

  friend matimpl::ConstituentItemBase;

 private:

  //! Entité nulle
  static ComponentItemInternal nullComponentItemInternal;

 public:

  ComponentItemInternal() = default;

 public:

  //! Indexeur dans les variables matériaux
  ARCCORE_HOST_DEVICE MatVarIndex variableIndex() const
  {
    return m_shared_info->_varIndex(m_component_item_internal_local_id);
  }

  //! Identifiant du composant
  ARCCORE_HOST_DEVICE Int32 componentId() const { return m_shared_info->_componentId(m_component_item_internal_local_id); }

  //! Indique s'il s'agit de la maille nulle.
  ARCCORE_HOST_DEVICE constexpr bool null() const { return m_component_item_internal_local_id.isNull(); }

  /*!
   * \brief Composant associé.
   *
   * Cet appel n'est valide que pour les mailles matériaux ou milieux. Si on souhaite
   * un appel valide pour toutes les 'ComponentItem', il faut utiliser componentId().
   */
  IMeshComponent* component() const { return m_shared_info->_component(m_component_item_internal_local_id); }

  //! Nombre de sous-composants.
  ARCCORE_HOST_DEVICE Int32 nbSubItem() const
  {
    return m_shared_info->_nbSubConstituent(m_component_item_internal_local_id);
  }

  //! Entité globale correspondante.
  impl::ItemBase globalItemBase() const
  {
    return m_shared_info->_globalItemBase(m_component_item_internal_local_id);
  }

  ARCCORE_HOST_DEVICE constexpr Int32 level() const { return m_shared_info->m_level; }

  //! Numéro unique de l'entité component
  Int64 componentUniqueId() const
  {
    return m_shared_info->_componentUniqueId(m_component_item_internal_local_id);
  }

 protected:

  // NOTE : Cette classe est partagée avec le wrapper C#
  // Toute modification de la structure interne doit être reportée
  // dans la structure C# correspondante
  //MatVarIndex m_var_index;
  ComponentItemInternalLocalId m_component_item_internal_local_id;
  ComponentItemSharedInfo* m_shared_info = nullptr;

 private:

  //! Entité nulle
  static ComponentItemInternal* _nullItem()
  {
    return &nullComponentItemInternal;
  }

  //! Composant supérieur (null si aucun)
  inline matimpl::ConstituentItemBase _superItemBase() const;

  //! Première entité du sous-constituant
  ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _firstSubItemLocalId() const
  {
    return m_shared_info->_firstSubConstituentLocalId(m_component_item_internal_local_id);
  }

  ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 i) const;

  ARCCORE_HOST_DEVICE ComponentItemInternalLocalId _internalLocalId() const
  {
    return m_component_item_internal_local_id;
  }

  void _reset(ComponentItemInternalLocalId id, ComponentItemSharedInfo* shared_info)
  {
    m_component_item_internal_local_id = id;
    m_shared_info = shared_info;
    m_shared_info->_reset(id);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ComponentItemInternal* ComponentItemSharedInfo::
_itemInternal(ComponentItemInternalLocalId id)
{
  return m_component_item_internal_view.ptrAt(id.localId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr matimpl::ConstituentItemBase::
ConstituentItemBase(ComponentItemSharedInfo* shared_info, ComponentItemInternalLocalId id)
: m_component_item(shared_info->_itemInternal(id))
{
}

inline ComponentItemSharedInfo* matimpl::ConstituentItemBase::
_sharedInfo() const
{
  return m_component_item->m_shared_info;
}

inline ComponentItemInternalLocalId matimpl::ConstituentItemBase::
_localId() const
{
  return m_component_item->m_component_item_internal_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr matimpl::ConstituentItemBase ComponentItemSharedInfo::
_item(ComponentItemInternalLocalId id)
{
  return matimpl::ConstituentItemBase(m_component_item_internal_view.ptrAt(id.localId()));
}

inline matimpl::ConstituentItemBase ComponentItemSharedInfo::
_superItemBase(ComponentItemInternalLocalId id) const
{
  ARCCORE_CHECK_RANGE(id.localId(), -1, m_storage_size);
  ComponentItemInternalLocalId super_local_id(m_super_component_item_local_id_data[id.localId()]);
  return m_super_component_item_shared_info->_item(super_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase ComponentItemInternal::
_subItemBase(Int32 i) const
{
  ComponentItemInternalLocalId lid(_firstSubItemLocalId().localId() + i);
  return m_shared_info->m_sub_component_item_shared_info->_item(lid);
}

matimpl::ConstituentItemBase ComponentItemInternal::
_superItemBase() const
{
  return m_shared_info->_superItemBase(m_component_item_internal_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ARCCORE_HOST_DEVICE MatVarIndex matimpl::ConstituentItemBase::
variableIndex() const
{
  return _sharedInfo()->_varIndex(_localId());
}

inline ARCCORE_HOST_DEVICE Int32 matimpl::ConstituentItemBase::
componentId() const
{
  return m_component_item->componentId();
}

inline ARCCORE_HOST_DEVICE constexpr bool matimpl::ConstituentItemBase::
null() const
{
  return m_component_item->null();
}

inline IMeshComponent* matimpl::ConstituentItemBase::
component() const
{
  return m_component_item->component();
}

inline ARCCORE_HOST_DEVICE Int32 matimpl::ConstituentItemBase::
nbSubItem() const
{
  return m_component_item->nbSubItem();
}

inline impl::ItemBase matimpl::ConstituentItemBase::
globalItemBase() const
{
  return m_component_item->globalItemBase();
}

inline ARCCORE_HOST_DEVICE constexpr Int32 matimpl::ConstituentItemBase::
level() const
{
  return m_component_item->level();
}

inline Int64 matimpl::ConstituentItemBase::
componentUniqueId() const
{
  return m_component_item->componentUniqueId();
}

inline void matimpl::ConstituentItemBase::
_setVariableIndex(MatVarIndex index)
{
  _sharedInfo()->_setVarIndex(_localId(), index);
}

inline matimpl::ConstituentItemBase matimpl::ConstituentItemBase::
_superItemBase() const
{
  return m_component_item->_superItemBase();
}

inline void matimpl::ConstituentItemBase::
_setSuperAndGlobalItem(ComponentItemInternalLocalId cii, ItemLocalId ii)
{
  _sharedInfo()->_setSuperItem(_localId(), cii);
  _sharedInfo()->_setGlobalItem(_localId(), ii);
}

inline void matimpl::ConstituentItemBase::
_setGlobalItem(ItemLocalId ii)
{
  _sharedInfo()->_setGlobalItem(_localId(), ii);
}

//! Première entité sous-composant.
inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId matimpl::ConstituentItemBase::
_firstSubItemLocalId() const
{
  return m_component_item->_firstSubItemLocalId();
}

inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase matimpl::ConstituentItemBase::
_subItemBase(Int32 i) const
{
  return m_component_item->_subItemBase(i);
}

//! Positionne le nombre de sous-composants.
inline void matimpl::ConstituentItemBase::
_setNbSubItem(Int16 nb_sub_item)
{
  _sharedInfo()->_setNbSubConstituent(_localId(), nb_sub_item);
}

//! Positionne le premier sous-composant.
inline void matimpl::ConstituentItemBase::
_setFirstSubItem(ComponentItemInternalLocalId first_sub_item)
{
  _sharedInfo()->_setFirstSubConstituentLocalId(_localId(), first_sub_item);
}

inline void matimpl::ConstituentItemBase::
_setComponent(Int16 component_id)
{
  _sharedInfo()->_setComponentId(_localId(), component_id);
}

inline ARCCORE_HOST_DEVICE ComponentItemInternalLocalId matimpl::ConstituentItemBase::
_internalLocalId() const
{
  return m_component_item->m_component_item_internal_local_id;
}

inline void matimpl::ConstituentItemBase::
_reset(ComponentItemInternalLocalId id, ComponentItemSharedInfo* shared_info)
{
  m_component_item->_reset(id, shared_info);
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
                                 ConstArrayView<ComponentItemInternalLocalId> ids,
                                 ConstArrayView<ComponentItemInternal*> items_internal)
  : m_component_shared_info(shared_info)
  , m_ids(ids)
  , m_items_internal(items_internal)
  {
#ifdef ARCANE_CHECK
    _checkCoherency();
#endif
  }

 private:

  matimpl::ConstituentItemBase _constituenItemBase(Int32 index) const
  {
#ifdef ARCANE_CHECK
    if (m_items_internal[index]->m_shared_info != m_component_shared_info)
      _throwIncoherentSharedInfo(index);
#endif
    return matimpl::ConstituentItemBase(m_items_internal[index]);
  }
  MatVarIndex _matVarIndex(Int32 index) const { return m_items_internal[index]->variableIndex(); }
  ConstituentItemLocalIdListView _subView(Int32 begin, Int32 size) const
  {
    return { m_component_shared_info, m_ids.subView(begin, size), m_items_internal.subView(begin, size) };
  }
  //! Pour les tests, vérifie que les vues pointent vers les mêmes données
  bool _isSamePointerData(const ConstituentItemLocalIdListView& rhs) const
  {
    return (m_ids.data() == rhs.m_ids.data()) && (m_items_internal.data() == rhs.m_items_internal.data());
  }
  friend bool operator==(const ConstituentItemLocalIdListView& a,
                         const ConstituentItemLocalIdListView& b)
  {
    bool t1 = a.m_component_shared_info == b.m_component_shared_info;
    bool t2 = a.m_ids == b.m_ids;
    bool t3 = a.m_items_internal == b.m_items_internal;
    return (t1 && t2 && t3);
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
  ConstArrayView<ComponentItemInternalLocalId> m_ids;
  ConstArrayView<ComponentItemInternal*> m_items_internal;

 private:

  void _checkCoherency() const;
  void _throwIncoherentSharedInfo(Int32 index) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
