// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInitialAllocator.h                                     (C) 2000-2023 */
/*                                                                           */
/* Interface d'allocation des entités du maillage.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHINITIALALLOCATOR_H
#define ARCANE_IMESHINITIALALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
 * \brief Allocateur pour les maillages polyédriques.
 */
class ItemAllocationInfo;
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

  virtual IUnstructuredMeshInitialAllocator* unstructuredMeshAllocator()
  {
    return nullptr;
  }

  virtual IPolyhedralMeshInitialAllocator* polyhedralMeshAllocator()
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

