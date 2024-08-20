// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.h                                   (C) 2000-2024 */
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
#include "arcane/core/ICartesianMeshGenerationInfo.h"

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
 * \deprecated Cette classe est obsolète. Il faut utiliser la version 2
 * de l'implémentation (CartesianMeshCoarsening2).
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
 * - createCoarseCells() qui créé les mailles grossières. Après appel à
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
 * ICartesianMesh* cartesian_mesh = ...;
 * Ref<CartesianMeshCoarsening> coarser = m_cartesian_mesh->createCartesianMeshCoarsening();
 * IMesh* mesh = cartesian_mesh->mesh();
 * CellInfoListView cells(mesh->cellFamily());
 * coarser->createCoarseCells();
 * Int32 index = 0;
 * for( Int32 cell_lid : coarser->coarseCells()){
 *   Cell cell = cells[cell_lid];
 *   info() << "Test: CoarseCell= " << ItemPrinter(cell);
 *   ConstArrayView<Int32> sub_cells(coarser->refinedCells(index));
 *   ++index;
 *   for( Int32 sub_lid : sub_cells )
 *     info() << "SubCell=" << ItemPrinter(cells[sub_lid]);
 * }
 * coarser->removeRefinedCells();
 * \endcode
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshCoarsening
: public TraceAccessor
{
  friend CartesianMeshImpl;

 private:

  ARCANE_DEPRECATED_REASON("Y2024: Use Arcane::CartesianMeshUtils::createCartesianMeshCoarsening2() instead")
  explicit CartesianMeshCoarsening(ICartesianMesh* m);

 public:

  /*!
   * \brief Déraffine le maillage initial par 2.
   *
   * Cette méthode est collective.
   */
  void createCoarseCells();

  /*!
   * \brief Liste des localIds() des mailles raffinées pour la maille parente \a d'indice \a index.
   *
   * Cette méthode n'est valide qu'après appel à createCoarseCells().
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
   * Cette méthode n'est valide qu'après appel à createCoarseCells().
   */
  ConstArrayView<Int32> coarseCells() const { return m_coarse_cells; }

  /*!
   * \brief Supprime les mailles raffinées.
   *
   * Il faut avoir appeler createCoarseCells() avant.
   */
  void removeRefinedCells();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  Int32 m_verbosity_level = false;
  UniqueArray2<Int32> m_refined_cells;
  UniqueArray<Int32> m_coarse_cells;
  bool m_is_create_coarse_called = false;
  bool m_is_remove_refined_called = false;
  Int64 m_first_own_cell_unique_id_offset = NULL_ITEM_UNIQUE_ID;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
  void _recomputeMeshGenerationInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

