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
#include "arcane/utils/Array2.h"

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
 * Le maillage initial doit être cartésien et ne doit pas avoir de patchs.
 *
 * Le maillage doit être un maillage AMR (IMesh::isAmrActivated()==true).
 *
 * Le nombre de mailles dans chaque dimension doit être un multiple de 2
 * ainsi que le nombre de mailles locales à chaque sous-domaine.
 *
 * Le dé-raffinement se fait en deux phases:
 *
 * - coarseCartesianMesh() qui créé les mailles grossières. Après appel à
 *   cette méthode il est possible d'utiliser coarseCells() pour avoir
 *   la liste des mailles grossières et refinedCells() pour avoir pour
 *   chaque maille grossière la liste des mailles raffinées correspondantes.
 * - removeRefinedCells() qui supprime les mailles autres que les mailles
 *   grossière. Après cet appel, il n'y a plus qu'un maillage cartésien
 *   avec 2 fois moins de mailles dans chaque direction. Il sera ensuite
 *   possible d'appeler les méthodes de raffinement pour créer des niveaux
 *   supplémentaires.
 *
 * \code
 * \endcode
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

  /*!
   * \brief Liste des localIds() des mailles raffinées pour la maille parente \a d'indice \a index.
   *
   * Cette méthode n'est valide qu'après appel à coarseCartesianMesh().
   *
   * En 2D, il y a 4 mailles raffinées par maille grossière. En 3D il y en a 8.
   */
  ConstArrayView<Int32> refinedCells(Int32 index) const
  {
    return m_refined_cells[index];
  }
  /*!
   * \brief Liste des localIds() des mailles grossières.
   *
   * Cette méthode n'est valide qu'après appel à coarseCartesianMesh().
   */
  ConstArrayView<Int32> coarseCells() const { return m_coarse_cells; }

  /*!
   * \brief Supprime les mailles raffinées.
   *
   * Il faut avoir appeler coarseCartesianMesh() avant.
   */
  void removeRefinedCells();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  Int32 m_verbosity_level = false;
  UniqueArray2<Int32> m_refined_cells;
  UniqueArray<Int32> m_coarse_cells;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

