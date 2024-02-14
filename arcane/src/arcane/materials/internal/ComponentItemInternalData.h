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

  class Sentinel{};
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
 public:

  explicit ComponentItemInternalData(MeshMaterialMng* mm);

 public:

  //! Notification de la fin de création des milieux/matériaux
  void endCreate();

 public:

  //! Liste des AllEnvCell
  ConstArrayView<ComponentItemInternal> allEnvItemsInternal() const
  {
    return m_all_env_items_internal;
  }

  //! Liste des AllEnvCell
  ArrayView<ComponentItemInternal> allEnvItemsInternal()
  {
    return m_all_env_items_internal;
  }

  //! Liste des mailles milieux.
  ArrayView<ComponentItemInternal> envItemsInternal()
  {
    return m_env_items_internal;
  }

  //! Liste des mailles matériaux pour le \a env_index ème milieu
  ArrayView<ComponentItemInternal> matItemsInternal(Int32 env_index)
  {
    return m_mat_items_internal[env_index];
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

  //! Redimensionne les structures allouant les 'ComponentItemInternal'
  void resizeComponentItemInternals(Int32 max_local_id, Int32 total_env_cell);

  //! Instance partagée associée au niveau \a level
  ComponentItemSharedInfo* sharedInfo(Int16 level) { return &m_shared_infos[level]; }
  ComponentItemSharedInfo* allEnvSharedInfo() { return sharedInfo(LEVEL_ALLENVIRONMENT); }
  ComponentItemSharedInfo* envSharedInfo() { return sharedInfo(LEVEL_ENVIRONMENT); }
  ComponentItemSharedInfo* matSharedInfo() { return sharedInfo(LEVEL_MATERIAL); }

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  UniqueArray<ComponentItemInternal> m_component_item_internal_storage;
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

 private:

  void
  _initSharedInfos();
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
