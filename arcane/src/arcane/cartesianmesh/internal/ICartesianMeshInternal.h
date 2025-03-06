// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshInternal.h                                    (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de ICartesianMesh.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
#define ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/ICartesianMeshAMRPatchMng.h"
#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

#include "arcane/core/ItemTypes.h"

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
   * \brief Méthode permettant de créer une instance de CartesianMeshAMRPatchMng.
   */
  virtual void initCartesianMeshAMRPatchMng() = 0;

  /*!
   * \brief Méthode permettant de récupérer l'instance de CartesianMeshAMRPatchMng.
   */
  virtual Ref<ICartesianMeshAMRPatchMng> cartesianMeshAMRPatchMng() = 0;

  // TODO
  virtual void initCartesianMeshNumberingMng() = 0;

  //TODO
  virtual Ref<ICartesianMeshNumberingMng> cartesianMeshNumberingMng() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
