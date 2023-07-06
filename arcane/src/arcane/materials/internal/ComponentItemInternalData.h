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

  ComponentItemInternalData(MeshMaterialMng* mm);

 public:

  //! Notification de la fin de création des milieux/matériaux
  void endCreate();

 public:

  //! Liste des mailles matériaux pour le \a env_index ème milieu
  ConstArrayView<ComponentItemInternal> matItemsInternal(Int32 env_index) const
  {
    return m_mat_items_internal[env_index];
  }

  //! Liste des mailles matériaux pour le \a env_index ème milieu
  ArrayView<ComponentItemInternal> matItemsInternal(Int32 env_index)
  {
    return m_mat_items_internal[env_index];
  }

  //! Redimensionne le nombre de mailles du \a env_index ème milieu.
  void resizeNbCellForEnvironment(Int32 env_index,Int32 size)
  {
    m_mat_items_internal[env_index].resize(size);;
  }

 private:

  MeshMaterialMng* m_material_mng = nullptr;

  //! Liste des ComponentItemInternal pour les matériaux de chaque milieu
  UniqueArray< UniqueArray<ComponentItemInternal> > m_mat_items_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
