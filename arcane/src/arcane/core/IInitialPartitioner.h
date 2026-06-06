// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IInitialPartitioner.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface of an initial partitioner.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IINITIALPARTITIONER_H
#define ARCANE_CORE_IINITIALPARTITIONER_H
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
 * \brief Interface of an initial partitioner.
 *
 * The service implementing this interface is responsible for the
 * initial partitioning of the case meshes. This partitioning takes place
 * only when the case starts, just before the case initialization.
 */
class ARCANE_CORE_EXPORT IInitialPartitioner
{
 public:

  virtual ~IInitialPartitioner() = default; //!< Releases resources.

 public:

  virtual void build() = 0;

 public:

  /*!
   * \brief Partitions the meshes.
   *
   * This operation must partition all \a meshes and
   * distribute them across all processors.
   */
  virtual void partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
