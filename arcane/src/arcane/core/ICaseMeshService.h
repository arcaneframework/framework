// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshService.h                                          (C) 2000-2024 */
/*                                                                           */
/* Interface of the service managing a dataset mesh.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMESHSERVICE_H
#define ARCANE_CORE_ICASEMESHSERVICE_H
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
 * \brief Interface of the service managing dataset meshes.
 *
 * Creation and initialization happen in 3 phases:
 * - a first phase when calling createMesh() where the mesh is created. In
 *   this phase, only the class managing the mesh is created, but they are not
 *   yet usable.
 * - a second phase when calling allocateMeshItems() where the mesh is
 *   allocated and initialized. This corresponds either to reading the mesh
 *   or dynamically creating the entities.
 * - a third phase which consists of partitioning the mesh if the code runs in
 *   parallel.
 * - a fourth phase that allows processing on the created mesh, such as a
 *   sub-division.
 */
class ICaseMeshService
{
 public:

  virtual ~ICaseMeshService() = default;

 public:

  //! Creates the mesh with the name \a name
  virtual void createMesh(const String& name) = 0;

  //! Allocates the mesh items
  virtual void allocateMeshItems() = 0;

  //! Partitions the mesh.
  virtual void partitionMesh() = 0;

  //! Applies operations after everything else.
  virtual void applyAdditionalOperations() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
