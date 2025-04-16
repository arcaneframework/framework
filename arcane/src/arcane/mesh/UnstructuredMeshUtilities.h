// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshUtilities.h                                 (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires sur un maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_UNSTRUCTUREDMESHUTILITIES_H
#define ARCANE_MESH_UNSTRUCTUREDMESHUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IMeshUtilities.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class ItemPairGroup;
class ItemPairGroupImpl;
class BasicItemPairGroupComputeFunctor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires sur un maillage.
 */
class UnstructuredMeshUtilities
: public TraceAccessor
, public IMeshUtilities
{
 public:

  explicit UnstructuredMeshUtilities(IMesh* mesh);
  ~UnstructuredMeshUtilities() override; //!< Libère les ressources.

 public:

  void changeOwnersFromCells() override;

  void localIdsFromConnectivity(eItemKind item_kind,
                                IntegerConstArrayView items_nb_node,
                                Int64ConstArrayView items_connectivity,
                                Int32ArrayView local_ids,
                                bool allow_null) override;
  void getFacesLocalIdFromConnectivity(ConstArrayView<ItemTypeId> items_type,
                                       ConstArrayView<Int64> items_connectivity,
                                       ArrayView<Int32> local_ids,
                                       bool allow_null) override;

  /*!
   * \brief Calcule la normale d'un groupe de face.
   *
   * Cette méthode calcule la normale à un groupe de face en considérant que
   * cette surface est un plan. Pour le calcul, l'algorithme essaie de
   * déterminer les noeuds aux extrémités de cette surface, et calcule une
   * normale à partir de ces noeuds. L'orientation de la normale est indéfinie.
   *
   * Si la surface n'est pas plane, le résultat est indéfini.
   *
   * L'algorithme actuel ne fonctionne pas toujours sur une surface composée
   * uniquement de triangles.
   *
   * Cette méthode est collective. L'algorithme utilisé garantit les
   * mêmes résultats en séquentiel et en parallèle.
   *
   * La variable \a nodes_coord est utilisée comme coordonnées pour les noeuds.
   * En général, il s'agit de IMesh::nodesCoordinates().
   */
  Real3 computeNormal(const FaceGroup& face_group,
                      const VariableNodeReal3& nodes_coord) override;

  Real3 computeDirection(const NodeGroup& node_group,
                         const VariableNodeReal3& nodes_coord,
                         Real3* n1, Real3* n2) override;

  void computeAdjency(ItemPairGroup adjency_array, eItemKind link_kind,
                      Integer nb_layer) override;

  bool writeToFile(const String& file_name, const String& service_name) override;

  void partitionAndExchangeMeshWithReplication(IMeshPartitionerBase* partitioner,
                                               bool initial_partition) override;

  void mergeNodes(Int32ConstArrayView nodes_local_id,
                  Int32ConstArrayView nodes_to_merge_local_id,
                  bool allow_non_corresponding_face) override;

  void computeAndSetOwnersForNodes() override;
  void computeAndSetOwnersForFaces() override;

  void recomputeItemsUniqueIdFromNodesUniqueId() override;

 private:

  IMesh* m_mesh = nullptr;
  BasicItemPairGroupComputeFunctor* m_compute_adjacency_functor = nullptr;

 private:

  Real3 _round(Real3 value);
  Real3 _broadcastFarthestNode(Real distance, const Node& node,
                               const VariableNodeReal3& nodes_coord);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

