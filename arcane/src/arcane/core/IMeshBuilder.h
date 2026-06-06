// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshReader.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a mesh creation/reading service.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHBUILDER_H
#define ARCANE_CORE_IMESHBUILDER_H
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
 * \ingroup StandardService
 * \brief Interface of a mesh creation/reading service.
 */
class ARCANE_CORE_EXPORT IMeshBuilder
{
 public:

  virtual ~IMeshBuilder() = default; //<! Releases resources

 public:

  /*!
   * \brief Fills \a build_info with the necessary information to
   * create the mesh.
   *
   * Some values may be filled by the caller, but the instance
   * may optionally override them. In particular, it is possible
   * to specify the mesh factory to use.
   */
  virtual void fillMeshBuildInfo(MeshBuildInfo& build_info) =0;

  //! Allocates the mesh entities managed by this service.
  virtual void allocateMeshItems(IPrimaryMesh* pm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
