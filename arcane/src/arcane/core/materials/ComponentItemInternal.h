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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCANE_CORE_EXPORT ComponentItemInternalLocalId
{
 public:

  ComponentItemInternalLocalId() = default;
  explicit ComponentItemInternalLocalId(Int32 id)
  : m_id(id)
  {}
  Int32 localId() const { return m_id; }

 private:

  Int32 m_id = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ComponentItemSharedInfo
{
  friend class ComponentItemInternal;
  friend class ComponentItemInternalData;

 private:

  //! Pour l'entité nulle
  static ComponentItemSharedInfo null_shared_info;
  static ComponentItemSharedInfo* null_shared_info_pointer;
  static ComponentItemSharedInfo* _nullInstance() { return null_shared_info_pointer; }

 private:

  // NOTE : Cette classe est partagée avec le wrapper C#
  // Toute modification de la structure interne doit être reportée
  // dans la structure C# correspondante
  ItemSharedInfo* m_item_shared_info = ItemSharedInfo::nullInstance();
  Int16 m_level = (-1);
  ConstArrayView<IMeshComponent*> m_components;
  ComponentItemSharedInfo* m_parent_component_item_shared_info = null_shared_info_pointer;
  ArrayView<ComponentItemInternal> m_component_item_internal_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace matimpl
{
  class ARCANE_CORE_EXPORT ConstituentItemBase
  {
    friend Arcane::Materials::ComponentCell;
    friend Arcane::Materials::AllEnvCell;
    friend Arcane::Materials::EnvCell;
    friend Arcane::Materials::MatCell;

   public:

    ARCCORE_HOST_DEVICE constexpr ConstituentItemBase(ComponentItemInternal* component_item)
    : m_component_item(component_item)
    {
    }

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

    ARCCORE_HOST_DEVICE constexpr ComponentItemInternal* _internal() const { return m_component_item; }

   private:

    ComponentItemInternal* m_component_item = nullptr;
  };
} // namespace matimpl

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

 private:

  static const int MAT_INDEX_OFFSET = 10;

  //! Entité nulle
  static ComponentItemInternal nullComponentItemInternal;

 public:

  ComponentItemInternal()
  {
    m_var_index.reset();
  }

 public:

  //! Indexeur dans les variables matériaux
  ARCCORE_HOST_DEVICE constexpr MatVarIndex variableIndex() const
  {
    return m_var_index;
  }

  //! Identifiant du composant
  ARCCORE_HOST_DEVICE constexpr Int32 componentId() const { return m_component_id; }

  //! Indique s'il s'agit de la maille nulle.
  ARCCORE_HOST_DEVICE constexpr bool null() const { return m_var_index.null(); }

  /*!
   * \brief Composant associé.
   *
   * Cet appel n'est valide que pour les mailles matériaux ou milieux. Si on souhaite
   * un appel valide pour toutes les 'ComponentItem', il faut utiliser componentId().
   */
  IMeshComponent* component() const { return m_shared_info->m_components[m_component_id]; }

  //! Nombre de sous-composants.
  ARCCORE_HOST_DEVICE constexpr Int32 nbSubItem() const
  {
    return m_nb_sub_component_item;
  }

  //! Entité globale correspondante.
  impl::ItemBase globalItemBase() const
  {
    return impl::ItemBase(m_global_item_local_id, m_shared_info->m_item_shared_info);
  }

  ARCCORE_HOST_DEVICE constexpr Int32 level() const { return m_shared_info->m_level; }

  //! Numéro unique de l'entité component
  Int64 componentUniqueId() const
  {
    // TODO: Vérifier que arrayIndex() ne dépasse pas (1<<MAT_INDEX_OFFSET)
    impl::ItemBase item_base(globalItemBase());
    return (Int64)m_var_index.arrayIndex() + ((Int64)item_base.uniqueId() << MAT_INDEX_OFFSET);
  }

 protected:

  // NOTE : Cette classe est partagée avec le wrapper C#
  // Toute modification de la structure interne doit être reportée
  // dans la structure C# correspondante
  MatVarIndex m_var_index;
  Int16 m_component_id = -1;
  Int16 m_nb_sub_component_item = 0;
  Int32 m_global_item_local_id = NULL_ITEM_LOCAL_ID;
  ComponentItemInternalLocalId m_component_item_internal_local_id;
  ComponentItemInternalLocalId m_super_component_item_local_id;
  ComponentItemInternal* m_first_sub_component_item = nullptr;
  ComponentItemSharedInfo* m_shared_info = nullptr;

 private:

  void _checkIsInt16(Int32 v)
  {
    if (v < (-32768) || v > 32767)
      _throwBadCast(v);
  }
  void _throwBadCast(Int32 v);

 private:

  //! Entité nulle
  static ComponentItemInternal* _nullItem()
  {
    return &nullComponentItemInternal;
  }

  //! Positionne l'indexeur dans les variables matériaux.
  void _setVariableIndex(MatVarIndex index)
  {
    m_var_index = index;
  }

  //! Composant supérieur (0 si aucun)
  matimpl::ConstituentItemBase _superItemBase() const
  {
    return &m_shared_info->m_component_item_internal_view[m_super_component_item_local_id.localId()];
  }

  void _setSuperAndGlobalItem(ComponentItemInternal* cii, ItemLocalId ii)
  {
    if (cii)
      m_super_component_item_local_id = cii->_internalLocalId();
    m_global_item_local_id = ii.localId();
  }

  void _setGlobalItem(ItemLocalId ii)
  {
    m_global_item_local_id = ii.localId();
  }

  //! Première entité sous-composant.
  ARCCORE_HOST_DEVICE ComponentItemInternal* _firstSubItem() const
  {
    return m_first_sub_component_item;
  }

  //! Positionne le nombre de sous-composants.
  void _setNbSubItem(Int32 nb_sub_item)
  {
#ifdef ARCANE_CHECK
    _checkIsInt16(nb_sub_item);
#endif
    m_nb_sub_component_item = static_cast<Int16>(nb_sub_item);
  }

  //! Positionne le premier sous-composant.
  void _setFirstSubItem(ComponentItemInternal* first_sub_item)
  {
    m_first_sub_component_item = first_sub_item;
  }

  void _setComponent(Int32 component_id)
  {
#ifdef ARCANE_CHECK
    _checkIsInt16(component_id);
#endif
    m_component_id = static_cast<Int16>(component_id);
  }

  ComponentItemInternalLocalId _internalLocalId() const
  {
    return m_component_item_internal_local_id;
  }

  void _reset(ComponentItemInternalLocalId id, ComponentItemSharedInfo* shared_info)
  {
    m_var_index.reset();
    m_component_id = -1;
    m_super_component_item_local_id = {};
    m_component_item_internal_local_id = id;
    m_nb_sub_component_item = 0;
    m_first_sub_component_item = nullptr;
    m_global_item_local_id = NULL_ITEM_LOCAL_ID;
    m_shared_info = shared_info;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
