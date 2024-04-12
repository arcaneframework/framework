// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshInternal.h                                    (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de ICartesianMesh.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
#define ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/cartesianmesh/ICartesianMeshAMRPatchMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de ICartesianMesh.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshInternal
{
 public:

  virtual ~ICartesianMeshInternal() = default;

 public:

  /*!
   * \brief Créé une instance pour gérer le déraffinement du maillage (V2).
   * \warning Experimental method !
   */
  virtual Ref<CartesianMeshCoarsening2> createCartesianMeshCoarsening2() = 0;

  /*!
   * \brief Créé un patch avec tous les enfants des mailles \a parent_cells_local_id
   *
   * \a parent_cells_local_id est la liste des localId() des mailles parentes.
   * Les mailles filles de \a parent_cells doivent déjà avoir été créées.
   */
  virtual void addPatchFromExistingChildren(ConstArrayView<Int32> parent_cells_local_id) = 0;

  /*!
   * \brief TODO
   */
  virtual void initCartesianMeshAMRPatchMng() = 0;

  virtual Ref<ICartesianMeshAMRPatchMng> cartesianMeshAMRPatchMng() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
