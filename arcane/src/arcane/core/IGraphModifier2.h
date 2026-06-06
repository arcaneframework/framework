// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraphModifier2.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface for a mesh graph modification tool                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGRAPHMODIFIER2_H
#define ARCANE_CORE_IGRAPHMODIFIER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a mesh graph.
 */
class ARCANE_CORE_EXPORT IGraphModifier2
{
 public:

  virtual ~IGraphModifier2() = default; //!< Frees resources

 public:

  //! Adds links to the graph with a fixed number of dual nodes per link
  virtual void addLinks(Integer nb_link,
                        Integer nb_dual_nodes_per_link,
                        Int64ConstArrayView links_infos) = 0;

  //! Adds dual nodes to the graph with a fixed dual item type per node
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Integer dual_node_kind,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Adds dual nodes to the graph, where the node type is specified in the infos array
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Removes dual nodes from the graph
  virtual void removeDualNodes(Int32ConstArrayView dual_node_local_ids) = 0;

  //! Removes dual links from the graph
  virtual void removeLinks(Int32ConstArrayView link_local_ids) = 0;

  //! Removes DualNodes and Links connected to cells that are being deleted
  virtual void removeConnectedItemsFromCells(Int32ConstArrayView cell_local_ids) = 0;

  virtual void endUpdate() = 0;

  virtual void updateAfterMeshChanged() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
