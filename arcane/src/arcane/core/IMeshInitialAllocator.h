// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInitialAllocator.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface d'allocation des entités du maillage.                           */
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
 * \brief Allocateur pour les maillages non structurés.
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
 * \brief Allocateur pour les maillages cartésiens.
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
 * \brief Allocateur pour les maillages polyédriques.
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
 * \brief Interface d'allocation des entités du maillage.
 */
class ARCANE_CORE_EXPORT IMeshInitialAllocator
{
 public:

  virtual ~IMeshInitialAllocator() = default;

 public:

  //! Allocateur pour les maillages non structurés
  virtual IUnstructuredMeshInitialAllocator* unstructuredMeshAllocator()
  {
    return nullptr;
  }

  //! Allocateur pour les maillages polyédriques
  virtual IPolyhedralMeshInitialAllocator* polyhedralMeshAllocator()
  {
    return nullptr;
  }

  //! Allocateur pour les maillages cartésiens
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

