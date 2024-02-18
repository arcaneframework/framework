// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternalData.h                                 (C) 2000-2024 */
/*                                                                           */
/* Gestion des listes de 'ComponentItemInternal'.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTITEMINTERNALDATA_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTITEMINTERNALDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/core/materials/ComponentItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interval des identifiants des constituants dans la liste des
 * ComponentItemInternal.
 */
class ComponentItemInternalRange
{
 public:

  class Sentinel
  {};
  class Iterator
  {
    friend ComponentItemInternalRange;

   private:

    Iterator(Int32 current_value, Int32 last_value)
    : m_current_value(current_value)
    , m_last_value(last_value)
    {}

   public:

    ComponentItemInternalLocalId operator*() const { return ComponentItemInternalLocalId(m_current_value); }
    void operator++() { ++m_current_value; }
    bool operator==(const Sentinel&) const
    {
      return m_current_value == m_last_value;
    }
    bool operator!=(const Sentinel&) const
    {
      return m_current_value != m_last_value;
    }

   private:

    Int32 m_current_value = 0;
    Int32 m_last_value = 0;
  };

 public:

  ComponentItemInternalLocalId operator[](Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_nb_value);
    return ComponentItemInternalLocalId(m_first_index + index);
  }

 public:

  void setRange(Int32 first_index, Int32 nb_value)
  {
    m_first_index = first_index;
    m_nb_value = nb_value;
  }
  Iterator begin() const
  {
    return Iterator(m_first_index, m_first_index + m_nb_value);
  }
  Sentinel end() const
  {
    return {};
  }
  Int32 size() const { return m_nb_value; }
  ComponentItemInternalLocalId firstValue() const { return ComponentItemInternalLocalId(m_first_index); }

 private:

  Int32 m_first_index = -1;
  Int32 m_nb_value = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des listes de 'ComponentItemInternal'.
 *
 * Il faut appeler endCreate() avant de pouvoir utiliser les instances de
 * cette classe.
 */
class ComponentItemInternalData
: public TraceAccessor
{
  //! Conteneur pour les informations de ComponentItemSharedInfo
  class Storage
  {
   public:

    explicit Storage(const MemoryAllocationOptions& alloc_info);

   public:

    void resize(Int32 new_size, ComponentItemSharedInfo* shared_info);

   private:

    UniqueArray<ComponentItemInternalLocalId> m_first_sub_constituent_item_id_list;
    UniqueArray<Int16> m_component_id_list;
    UniqueArray<Int16> m_nb_sub_constituent_item_list;
  };

 public:

  explicit ComponentItemInternalData(MeshMaterialMng* mm);

 public:

  //! Notification de la fin de création des milieux/matériaux
  void endCreate();

 public:

  //! Liste des AllEnvCell. Uniquement pour les itérateurs.
  ConstArrayView<ComponentItemInternal> allEnvItemsInternal() const
  {
    return m_all_env_items_internal;
  }

  //! Retourne la AllEnvCell correspondant à la maille \a id
  matimpl::ConstituentItemBase allEnvItemBase(CellLocalId id)
  {
    return matimpl::ConstituentItemBase(&m_all_env_items_internal[id.localId()]);
  }

  //! Retourne la EnvCell correspondant à l'indice \a index
  matimpl::ConstituentItemBase envItemBase(Int32 index)
  {
    return matimpl::ConstituentItemBase(&m_env_items_internal[index]);
  }

  //! Retourne la MatCell correspondant au milieu d'indice \a index du milieu \a env_index
  matimpl::ConstituentItemBase matItemBase(Int16 env_index,Int32 index)
  {
    return matimpl::ConstituentItemBase(&m_mat_items_internal[env_index][index]);
  }

  ComponentItemInternalRange allEnvItemsInternalRange() const
  {
    return m_all_env_items_internal_range;
  }

  //! Liste des mailles milieux.
  ComponentItemInternalRange envItemsInternalRange() const
  {
    return m_env_items_internal_range;
  }

  //! Liste des mailles matériaux pour le \a env_index ème milieu
  ComponentItemInternalRange matItemsInternalRange(Int32 env_index)
  {
    return m_mat_items_internal_range[env_index];
  }

  //! Redimensionne les structures allouant les 'ComponentItemInternal'
  void resizeComponentItemInternals(Int32 max_local_id, Int32 total_env_cell);

  //! Instance partagée associée au niveau \a level
  ComponentItemSharedInfo* sharedInfo(Int16 level) { return &m_shared_infos[level]; }
  ComponentItemSharedInfo* allEnvSharedInfo() { return sharedInfo(LEVEL_ALLENVIRONMENT); }
  ComponentItemSharedInfo* envSharedInfo() { return sharedInfo(LEVEL_ENVIRONMENT); }
  ComponentItemSharedInfo* matSharedInfo() { return sharedInfo(LEVEL_MATERIAL); }

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  UniqueArray<ComponentItemInternal> m_all_env_item_internal_storage;
  UniqueArray<ComponentItemInternal> m_env_item_internal_storage;
  UniqueArray<ComponentItemInternal> m_mat_item_internal_storage;
  /*!
   * \brief Liste des ComponentItemInternal pour les AllEnvcell.
   *
   * Les éléments de ce tableau peuvent être indexés directement avec
   * le localId() de la maille.
   */
  ArrayView<ComponentItemInternal> m_all_env_items_internal;

  //! Liste des ComponentItemInternal pour chaque milieu
  ArrayView<ComponentItemInternal> m_env_items_internal;

  //! Liste des ComponentItemInternal pour les matériaux de chaque milieu
  UniqueArray<ArrayView<ComponentItemInternal>> m_mat_items_internal;

  //! Liste des informations partagées
  UniqueArray<ComponentItemSharedInfo> m_shared_infos;

  ComponentItemInternalRange m_all_env_items_internal_range;
  ComponentItemInternalRange m_env_items_internal_range;
  UniqueArray<ComponentItemInternalRange> m_mat_items_internal_range;

  Storage m_all_env_storage;
  Storage m_env_storage;
  Storage m_mat_storage;

 private:

  void _initSharedInfos();
  void _resetMatItemsInternal(Int32 env_index);
  //! Réinitialise les ComponentItemInternal associés aux EnvCell et AllEnvCell
  void _resetItemsInternal();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
