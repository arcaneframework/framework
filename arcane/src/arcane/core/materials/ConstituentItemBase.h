// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemBase.h                                       (C) 2000-2024 */
/*                                                                           */
/* Informations génériques sur une entité d'un constituant.                  */
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
 * \brief Informations génériques sur une entité d'un constituant.
 *
 * Cette classe est le pendant de ItemInternal pour la gestion des matériaux
 * et des milieux. Elle ne doit en principe pas être utilisée directement, sauf
 * par les classes de Arcane. Il vaut mieux utiliser les
 * classes ComponentCell,  MatCell, EnvCell ou AllEnvCell.
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
    return ((a.m_constituent_item_index == b.m_constituent_item_index) && (a.m_shared_info == b.m_shared_info));
  }
  ARCCORE_HOST_DEVICE constexpr friend bool
  operator!=(const ConstituentItemBase& a, const ConstituentItemBase& b)
  {
    return !(a == b);
  }

 private:

  //! Positionne l'indexeur dans les variables matériaux.
  ARCCORE_HOST_DEVICE inline void _setVariableIndex(MatVarIndex index)
  {
    m_shared_info->_setVarIndex(m_constituent_item_index, index);
  }

  //! Composant supérieur (0 si aucun)
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

  //! Première entité sous-composant.
  inline ARCCORE_HOST_DEVICE ConstituentItemIndex _firstSubItemLocalId() const
  {
    return m_shared_info->_firstSubConstituentLocalId(m_constituent_item_index);
  }

  inline ARCCORE_HOST_DEVICE matimpl::ConstituentItemBase _subItemBase(Int32 i) const;

  //! Positionne le nombre de sous-composants.
  ARCCORE_HOST_DEVICE void _setNbSubItem(Int16 nb_sub_item)
  {
    m_shared_info->_setNbSubConstituent(m_constituent_item_index, nb_sub_item);
  }

  //! Positionne le premier sous-composant.
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
  return matimpl::ConstituentItemBase(this,id);
}

ARCCORE_HOST_DEVICE inline matimpl::ConstituentItemBase ComponentItemSharedInfo::
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
