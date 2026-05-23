// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshUtilities.h                                 (C) 2000-2025 */
/*                                                                           */
/* Utility functions for a mesh.                                             */
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
 * \brief Utility functions for a mesh.
 */
class UnstructuredMeshUtilities
: public TraceAccessor
  , public IMeshUtilities
{
public:

  explicit UnstructuredMeshUtilities(IMesh* mesh);
  ~UnstructuredMeshUtilities() override; //!< Frees resources.

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
   * \brief Calculates the normal of a face group.
   *
   * This method calculates the normal of a face group assuming that
   * this surface is a plane. For the calculation, the algorithm tries to
   * determine the nodes at the ends of this surface, and calculates a
   * normal from these nodes. The orientation of the normal is undefined.
   *
   * If the surface is not planar, the result is undefined.
   *
   * The current algorithm does not always work on a surface composed
   * solely of triangles.
   *
   * This method is collective. The algorithm used guarantees the
   * same results in sequential and parallel modes.
   *
   * The variable \a nodes_coord is used as coordinates for the nodes.
   * Generally, it is IMesh::nodesCoordinates().
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
  void computeAndSetOwnersForEdges() override;
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
