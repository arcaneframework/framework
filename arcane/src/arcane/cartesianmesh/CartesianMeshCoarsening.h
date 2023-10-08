// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.h                                   (C) 2000-2023 */
/*                                                                           */
/* Déraffinement d'un maillage cartésien.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ItemTypes.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 *
 * \brief Déraffine un maillage cartésien par 2.
 *
 * \warning Cette méthode est expérimentale.
 *
 * Cette classe permet de déraffiner un maillage cartésien. Les instances
 * de cette classe sont créées via ICartesianMesh::createCartesianMeshCoarsening().
 *
 * Après utilisation, le maillage sera un maillage AMR et le maillage
 * initial sera un patch (ICartesianMeshPatch). Les mailles du maillage
 * initial seront des mailles de niveau 1.
 *
 * Le maillage initial doit être cartésien et ne doit pas avoir de patchs.
 *
 * Le maillage doit être un maillage AMR (IMesh::isAmrActivated()==true).
 *
 * Le nombre de mailles dans chaque dimension doit être un multiple de 2
 * ainsi que le nombre de mailles locales à chaque sous-domaine.
 *
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshCoarsening
: public TraceAccessor
{
  friend CartesianMeshImpl;

 private:

  explicit CartesianMeshCoarsening(ICartesianMesh* m);

 public:

  /*!
   * \brief Déraffine le maillage initial par 2.
   *
   * Cette méthode est collective.
   */
  void coarseCartesianMesh();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

