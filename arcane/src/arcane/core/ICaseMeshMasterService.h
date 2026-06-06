// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshMasterService.h                                    (C) 2000-2024 */
/*                                                                           */
/* Interface of the service managing the meshes of the dataset.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMESHMASTERSERVICE_H
#define ARCANE_CORE_ICASEMESHMASTERSERVICE_H
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
 * \internal
 * \brief Interface of the service managing the meshes of the dataset.
 *
 * This service manages the different mesh creation/reading services.
 *
 * This is done in two phases:
 * - a first phase when calling createMeshes() where all the
 *   meshes derived from the dataset are created. In this phase, only
 *   the class managing the meshes is created, but they are not yet
 *   usable.
 * - a second phase when calling allocateMeshes() where the meshes
 *   are actually allocated. This corresponds either to reading the meshes,
 *   or to the effective creation of the entities they manage.
 *
 * There is a third optional phase which is only performed
 * in parallel and consists of partitioning the meshes, via the call
 * to partitionMeshes().
 *
 * Finally, it is possible to apply an additional treatment at the end of mesh creation.
 * For example, it is possible to subdivide the current mesh. This is done by calling
 * applyAdditionalOperationsOnMeshes().
 */
class ICaseMeshMasterService
{
 public:

  virtual ~ICaseMeshMasterService() = default;

 public:

  //! Creates the meshes
  virtual void createMeshes() = 0;
  //! Creates the meshes
  virtual void allocateMeshes() = 0;
  //! Partitions the meshes
  virtual void partitionMeshes() = 0;
  //! Applies any additional operations on the created mesh.
  virtual void applyAdditionalOperationsOnMeshes() {}

 public:

  virtual ICaseOptions* _options() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
