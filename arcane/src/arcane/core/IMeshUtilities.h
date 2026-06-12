// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshUtilities.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface of a class providing utility functions on meshes.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHUTILITIES_H
#define ARCANE_CORE_IMESHUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a class providing utility functions on meshes.
 */
class ARCANE_CORE_EXPORT IMeshUtilities
{
 public:

  virtual ~IMeshUtilities() = default; //!< Frees resources.

 public:

  /*!
   * \brief Searches for the local IDs of entities based on their
   * connectivity.
   *
   * This method is only implemented for order 1 faces.
   *
   * \deprecated Use getFacesLocalIdFromConnectivity() instead.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use getFacesLocalIdFromConnectivity() instead")
  virtual void localIdsFromConnectivity(eItemKind item_kind,
                                        IntegerConstArrayView items_nb_node,
                                        Int64ConstArrayView items_connectivity,
                                        Int32ArrayView local_ids,
                                        bool allow_null = false) = 0;

  /*!
   * \brief Searches for the local IDs of faces based on their connectivity.
   *
   * Takes as input a list of entities described by the unique IDs
   * (Item::uniqueId()) of their nodes and searches for the local IDs
   * (Item::localId())
   * of these entities.
   *
   * \param items_type array of ItemTypeId of the entities
   * \param items_connectivity array containing the unique indices of the
   *                           entities' nodes.
   * \param local_ids in return, contains the local IDs of the entities.
   *                  The number of elements in \a local_ids must be equal to that
   *                  of \a items_nb_node.
   *
   * The array \a items_connectivity contains the node IDs of the faces,
   * listed consecutively. For example, if \c items_type[0]==IT_Triangle3 and
   * \c items_type[1]==IT_Quad4, then \a items_connectivity[0..2] will contain the
   * nodes of entity 0, and items_connectivity[3..6] those of entity 1.
   *
   * If \a allow_null is false, a fatal error is generated if
   * an entity is not found; otherwise, NULL_ITEM_LOCAL_ID is
   * returned for the corresponding entity.
   */
  virtual void getFacesLocalIdFromConnectivity(ConstArrayView<ItemTypeId> items_type,
                                               ConstArrayView<Int64> items_connectivity,
                                               ArrayView<Int32> local_ids,
                                               bool allow_null = false) = 0;

  /*!
   * \brief Calculates the normal of a face group.
   *
   * This method calculates the normal of a face group by assuming that
   * this surface is a plane. For the calculation, the algorithm attempts to
   * determine the nodes at the ends of this surface and calculates a
   * normal from these nodes. The orientation of the normal (inward
   * or outward) is undefined.
   *
   * If the surface is not planar, the result is undefined.
   *
   * The current algorithm does not always work on a surface composed
   * only of triangles.
   *
   * This method is collective. The algorithm used guarantees the
   * same results in sequential and parallel execution.
   *
   * The variable \a nodes_coord is used as coordinates for the nodes.
   * Generally, this is IMesh::nodesCoordinates().
   */
  virtual Real3 computeNormal(const FaceGroup& face_group,
                              const VariableNodeReal3& nodes_coord) = 0;

  /*!
   * \brief Calculates the direction vector of a line.
   *
   * This method calculates the direction vector of a group of nodes
   * by assuming that it forms a line. For the calculation, the algorithm attempts to
   * determine the nodes at the ends of this line and calculates a
   * vector from these nodes. The direction of the vector is undefined.
   *
   * If the group does not form a line, the result is undefined.
   *
   * This method is collective. The algorithm used guarantees the
   * same results in sequential and parallel execution.
   *
   * If \a n1 and \a n2 are not null, they will contain the extreme coordinates
   * from which the direction is calculated.
   *
   * The variable \a nodes_coord is used as coordinates for the nodes.
   * Generally, this is IMesh::nodesCoordinates().
   */
  virtual Real3 computeDirection(const NodeGroup& node_group,
                                 const VariableNodeReal3& nodes_coord,
                                 Real3* n1, Real3* n2) = 0;

  //! Calculates adjacencies, stored in \a adjacency_array
  ARCANE_DEPRECATED_REASON("Y2020: Use computeAdjacency() instead")
  virtual void computeAdjency(ItemPairGroup adjacency_array, eItemKind link_kind,
                              Integer nb_layer) = 0;

  //! Calculates adjacencies, stored in \a adjacency_array
  virtual void computeAdjacency(const ItemPairGroup& adjacency_array, eItemKind link_kind,
                                Integer nb_layer);

  /*!
   * \brief Positions the new owners of nodes, edges, and faces based
   * on the cells.
   *
   * Assuming the new owners of the cells are known (and synchronized),
   * it determines the new owners of other entities and synchronizes them.
   *
   * This method is collective.
   *
   * \note This method requires that the synchronization information be valid.
   * If you want to determine the owners of entities without prior
   * information, you must use computeAndSetOwnersForNodes()
   * or computeAndSetOwnersForFaces().
   */
  virtual void changeOwnersFromCells() = 0;

  /*!
   * \brief Determines the owners of the nodes.
   *
   * The determination is based on the owners of the cells.
   * There must be no ghost cell layers.
   *
   * This operation is collective.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use MeshUtils::computeAndSetOwnerForNodes() instead")
  virtual void computeAndSetOwnersForNodes() = 0;

  /*!
   * \brief Determines the owners of the edges.
   *
   * The determination is based on the owners of the cells.
   * There must be no ghost cell layers.
   *
   * This operation is collective.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use MeshUtils::computeAndSetOwnerForEdges() instead")
  virtual void computeAndSetOwnersForEdges() = 0;

  /*!
   * \brief Determines the owners of the faces.
   *
   * The determination is based on the owners of the cells.
   * There must be no ghost cell layers.
   *
   * This operation is collective.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use MeshUtils::computeAndSetOwnerForFaces() instead")
  virtual void computeAndSetOwnersForFaces() = 0;

  /*!
   * \brief Writes the mesh to a file.
   *
   * Writes the mesh to the file \a file_name using the service implementing
   * the 'IMeshWriter' interface and named \a service_name.
   *
   * \retval true if the specified service is not available.
   * \retval false if everything is ok.
   */
  virtual bool writeToFile(const String& file_name, const String& service_name) = 0;

  /*!
   * \brief Repartitions and exchanges the mesh while managing replication.
   *
   * This method performs a mesh repartitioning via
   * the call to IMeshPartitioner::partitionMesh(bool) and proceeds to exchange
   * entities via IPrimaryMesh::exchangeItems().
   *
   * It also manages replication by ensuring that all replicas
   * have the same mesh.
   * The principle is as follows:
   * - only the master replica performs the repartitioning by calling
   * IMeshPartitioner::partitionMesh() with \a partitioner as the partitioner
   * - the values of IItemFamily::itemsNewOwner() are then
   * synchronized with the other replicas.
   * - entity exchanges are performed via IPrimaryMesh::exchangeItems().
   *
   * This method is collective across all replicas.
   *
   * \pre All replicas must have the same mesh, meaning that all entity families
   * must be identical except for particle families which are not concerned.
   * \pre The mesh must be an instance of IPrimaryMesh.
   *
   * \post All replicas have the same mesh except for particle families.
   *
   * \param partitioner Instance of the partitioner to be used
   * \param initial_partition Indicates if it is the initial partitioning.
   */
  virtual void partitionAndExchangeMeshWithReplication(IMeshPartitionerBase* partitioner,
                                                       bool initial_partition) = 0;

  /*!
   * \brief Merges nodes.
   */
  virtual void mergeNodes(Int32ConstArrayView nodes_local_id,
                          Int32ConstArrayView nodes_to_merge_local_id)
  {
    this->mergeNodes(nodes_local_id, nodes_to_merge_local_id, false);
  }

  /*!
   * \brief Merges nodes.
   *
   * Merges nodes in pairs from \a nodes_to_merge_local_id with those
   * from \a nodes_local_id. Each node \a nodes_to_merge_local_id[i] is
   * merged with \a nodes_local_id[i].
   *
   * The nodes \a nodes_to_merge_local_id are destroyed after merging. Entities
   * entirely resting on these merged nodes are also destroyed.
   *
   * It is forbidden to merge two nodes from the same cell or the same face
   * (after merging, a face or a cell cannot have the same node twice).
   *
   * Once the merge is performed, the faces containing the merged nodes
   * (\a nodes_to_merge_local_id) are destroyed. If \a allow_non_corresponding_face
   * is false, then for each destroyed face, there must correspond an existing face
   * with the merged nodes (\a nodes_local_id).
   */
  virtual void mergeNodes(Int32ConstArrayView nodes_local_id,
                          Int32ConstArrayView nodes_to_merge_local_id,
                          bool allow_non_corresponding_face) = 0;
  /*!
   * \brief Recalculates the uniqueId() of edges, faces, and cells based on the
   * uniqueId() of the nodes.
   *
   * \warning This method is experimental and should only be used within Arcane.
   * It assumes that the uniqueId() of the entities are constructed from
   * generateHashUniqueId().
   */
  virtual void recomputeItemsUniqueIdFromNodesUniqueId() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
