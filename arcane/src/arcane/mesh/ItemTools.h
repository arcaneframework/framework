// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTools.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Utilities helping to find items based on others                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMTOOLS_H
#define ARCANE_MESH_ITEMTOOLS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Utilities helping to find items based on others
 */
class ItemTools
{
 public:

  /*!
   * Checks if the list of nodes of a face matches a provided list
   * Compares the uids of the nodes. The order must be the same.
   *
   * @param face : the face to test
   * @param face_nodes_uid : a list of node uids
   *
   */
  static bool isSameFace(Face face, Int64ConstArrayView face_nodes_uid);

  /*!
   * Searches for a face connected to the node \a node corresponding to the list of
   * nodes \a  face_nodes_uid.
   *
   * @param node : node to test
   * @param face_type_id : type of the face searched
   * @param face_nodes_uid : a list of node uids
   *
   */
  static Face findFaceInNode2(Node node,
                              Integer face_type_id,
                              Int64ConstArrayView face_nodes_uid);

  /*!
   * Searches for an edge connected to the node \a node and connecting the nodes 
   * with uids \a begin_node and \a end_node
   *
   * @param node : node to test
   * @param begin_node : uid of the first node of the searched edge
   * @param end_node : uid of the second node of the searched edge
   *
   */
  static Edge findEdgeInNode2(Node node, Int64 begin_node, Int64 end_node);

 private:

  /*!
   * Searches for a face connected to the node \a node corresponding to the list of
   * nodes \a  face_nodes_uid.
   *
   * @param node : node to test
   * @param face_type_id : type of the face searched
   * @param face_nodes_uid : a list of node uids
   *
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use findFaceInNode2() instead")
  static ItemInternal* findFaceInNode(Node node,
                                      Integer face_type_id,
                                      Int64ConstArrayView face_nodes_uid);

  /*!
   * Searches for an edge connected to the node \a node and connecting the nodes
   * with uids \a begin_node and \a end_node
   *
   * @param node : node to test
   * @param begin_node : uid of the first node of the searched edge
   * @param end_node : uid of the second node of the searched edge
   *
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use findEdgeInNode2() instead")
  static ItemInternal* findEdgeInNode(Node node, Int64 begin_node, Int64 end_node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
