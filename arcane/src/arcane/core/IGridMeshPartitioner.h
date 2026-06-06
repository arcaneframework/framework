// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGridMeshPartitioner.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh partitioner on a grid.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGRIDMESHPARTITIONER_H
#define ARCANE_CORE_IGRIDMESHPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshPartitionerBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a mesh partitioner on a grid.
 *
 * This partitioner redistributes the mesh in a 2D or 3D grid.
 *
 * The grid is composed of a set of parts, each part being
 * defined by its bounding box (the coordinates of the two
 * points at the ends of the grid) and an index (i,j,k). The
 * dimensions of each part can be different but they
 * must be consistent: all parts with the same index \a i
 * must have the same bounding box \a x coordinate.
 *
 * ------------------------
 * | 0,2 | 1,2   |2,2| 3,2 |
 * ------------------------
 * | 0,1 | 1,1   |2,1| 3,1 |
 * ------------------------
 * | 0,0 | 1,0   |2,0| 3,0 |
 * ------------------------
 *
 * Instances of this class are single-use and must only be used
 * for a single partitioning. The user must call setBoundingBox()
 * and setPartIndex() and then perform the partitioning by calling
 * applyMeshPartitioning().
 */
class ARCANE_CORE_EXPORT IGridMeshPartitioner
// TODO: supprimer l'héritage de cette interface
: public IMeshPartitionerBase
{
 public:

  /*!
   * \brief Positions the bounding box of our subdomain.
   *
   * For the algorithm to work, there must be no overlap
   * between the bounding boxes of the subdomains.
   */
  virtual void setBoundingBox(Real3 min_val, Real3 max_val) = 0;

  /*!
   * \brief Index (i,j,k) of the part.
   *
   * The indices start at zero. In 1D or 2D, the value of \a k is \a
   * (-1). In 1D, the value of \a j is \a (-1)
   */
  virtual void setPartIndex(Int32 i, Int32 j, Int32 k) = 0;

  /*!
   * \brief Applies the repartitioning to the mesh \a mesh.
   *
   * The setPartIndex() and setBoundingBox() methods must have been
   * called previously. This method can only be called once
   * per instance.
   *
   * The partitioning uses a specific algorithm for
   * calculating ghost cells to ensure that every cell in \a mesh
   * that intersects the part specified in setBoundingBox() will be in
   * this subdomain, possibly as a ghost cell.
   */
  virtual void applyMeshPartitioning(IMesh* mesh) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
