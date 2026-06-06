// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInitialAllocator.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface for allocating mesh entities.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHINITIALALLOCATOR_H
#define ARCANE_CORE_IMESHINITIALALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ItemAllocationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allocator for unstructured meshes.
 */
class ARCANE_CORE_EXPORT IUnstructuredMeshInitialAllocator
{
 public:

  virtual ~IUnstructuredMeshInitialAllocator() = default;

 public:

  virtual void allocate(UnstructuredMeshAllocateBuildInfo& build_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allocator for Cartesian meshes.
 */
class ARCANE_CORE_EXPORT ICartesianMeshInitialAllocator
{
 public:

  virtual ~ICartesianMeshInitialAllocator() = default;

 public:

  virtual void allocate(CartesianMeshAllocateBuildInfo& build_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allocator for polyhedral meshes.
 */
class ARCANE_CORE_EXPORT IPolyhedralMeshInitialAllocator
{
 public:

  virtual ~IPolyhedralMeshInitialAllocator() = default;

 public:

  virtual void allocateItems(const Arcane::ItemAllocationInfo& allocation_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for allocating mesh entities.
 */
class ARCANE_CORE_EXPORT IMeshInitialAllocator
{
 public:

  virtual ~IMeshInitialAllocator() = default;

 public:

  //! Allocator for unstructured meshes
  virtual IUnstructuredMeshInitialAllocator* unstructuredMeshAllocator()
  {
    return nullptr;
  }

  //! Allocator for polyhedral meshes
  virtual IPolyhedralMeshInitialAllocator* polyhedralMeshAllocator()
  {
    return nullptr;
  }

  //! Allocator for Cartesian meshes
  virtual ICartesianMeshInitialAllocator* cartesianMeshAllocator()
  {
    return nullptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
