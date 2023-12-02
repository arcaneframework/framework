// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternalData.h                                 (C) 2000-2023 */
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

  //! Redimensionne le nombre de AllEnvCell
  void resizeNbAllEnvCell(Int32 size)
  {
    m_all_env_items_internal.resize(size);
  }

  //! Redimensionne le nombre milieu
  void resizeNbEnvCell(Int32 size)
  {
    m_env_items_internal.resize(size);
  }

  //! Redimensionne le nombre de mailles matériaux du \a env_index- ème milieu.
  void resizeNbMatCellForEnvironment(Int32 env_index,Int32 size)
  {
    m_mat_items_internal[env_index].resize(size);
  }

  void resetEnvItemsInternal();

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  /*!
   * \brief Liste des ComponentItemInternal pour les AllEnvcell.
   *
   * Les éléments de ce tableau peuvent être indexés directement avec
   * le localId() de la maille.
   */
  UniqueArray<ComponentItemInternal> m_all_env_items_internal;

  //! Liste des ComponentItemInternal pour chaque milieu
  UniqueArray<ComponentItemInternal> m_env_items_internal;

  //! Liste des ComponentItemInternal pour les matériaux de chaque milieu
  UniqueArray< UniqueArray<ComponentItemInternal> > m_mat_items_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
