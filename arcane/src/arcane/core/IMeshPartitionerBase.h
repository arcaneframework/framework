// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitioner.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh partitioner.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONERBASE_H
#define ARCANE_CORE_IMESHPARTITIONERBASE_H
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
 * \brief Interface of a mesh partitioner.
 */
class IMeshPartitionerBase
{
 public:

  virtual ~IMeshPartitionerBase() = default; //!< Releases resources.

 public:

  /*!
   * Re-partitions the mesh \a mesh
   *
   * This method changes the owners of the entities and
   * fills the IItemFamily::itemsNewOwner() variable of each entity family
   * of the mesh \a mesh with the number of the new owning subdomain.
   *
   * \note This method is reserved for Arcane developers.
   * If a module wishes to perform a re-partitioning,
   * it must call the method 
   * IMeshUtilities::partitionAndExchangeMeshWithReplication()
   * which handles both the partitioning and the exchange of
   * information and supports domain replication.
   */
  virtual void partitionMesh(bool initial_partition) = 0;

  //! Associated mesh
  virtual IPrimaryMesh* primaryMesh() = 0;

  //! Notification when a re-partitioning finishes (after entity exchange)
  virtual void notifyEndPartition() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
