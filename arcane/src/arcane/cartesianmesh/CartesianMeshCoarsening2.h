// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening2.h                                  (C) 2000-2024 */
/*                                                                           */
/* Déraffinement d'un maillage cartésien.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING2_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING2_H
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
 * Le dé-raffinement se lors de l'appel à createCoarseCell(). Après cet
 * appel on a la structure suivante pour le maillage :
 * - des mailles grossières sont créées pour chaque quadruplet de mailles
 *   existantes qui deviennent donc des mailles raffinées.
 * - chaque maille grossière est de niveau 0 (Cell::level()) et chaque maille
 *   initiale est de niveau 1.
 * - on peut accéder à ces mailles filles via les méthodes Cell::nbHChildren() et
 *   Cell::hChild().
 * - il y aura deux patchs dans le maillage. Le premier contiendra les mailles
 *   de niveau zéro et le second contiendra les mailles de niveau 1 qui sont les
 *   anciennes mailles avant dé-raffinement.
 *
 * Il est ensuite possible de ne conserver que les mailles grossières et de
 * supprimer les mailles raffinées par la'appel à la méthode.
 *
 * - removeRefinedCells() qui supprime les mailles autres que les mailles
 *   grossière. Après cet appel, il n'y a plus qu'un maillage cartésien
 *   avec 2 fois moins de mailles dans chaque direction. Il sera ensuite
 *   possible d'appeler les méthodes de raffinement pour créer des niveaux
 *   supplémentaires.
 * Voici un exemple de code utilisateur:
 *
 * \code
 * ICartesianMesh* cartesian_mesh = ...;
 * Ref<CartesianMeshCoarsening> coarser = CartesianMeshUtils::createCartesianMeshCoarsening(cartesian_mesh);
 * coarser->createCoarseCells();
 * \endcode
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshCoarsening2
: public TraceAccessor
{
  friend CartesianMeshImpl;

 private:

  explicit CartesianMeshCoarsening2(ICartesianMesh* m);

 public:

  /*!
   * \brief Déraffine le maillage initial par 2.
   *
   * Cette méthode est collective.
   */
  void createCoarseCells();

  void removeRefinedCells();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  Int32 m_verbosity_level = false;
  //! uniqueId() des mailles grossières
  UniqueArray<Int64> m_coarse_cells_uid;
  bool m_is_create_coarse_called = false;
  bool m_is_remove_refined_called = false;
  Int64 m_first_own_cell_unique_id_offset = NULL_ITEM_UNIQUE_ID;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
  void _recomputeMeshGenerationInfo();
  void _writeMeshSVG(const String& name);
  void _doDoubleGhostLayers();
  void _createCoarseCells2D();
  void _createCoarseCells3D();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

