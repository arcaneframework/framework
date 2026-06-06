// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshFactoryMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of the mesh factory manager.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHFACTORYMNG_H
#define ARCANE_CORE_IMESHFACTORYMNG_H
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
 * \brief Interface of the mesh factory manager.
 */
class ARCANE_CORE_EXPORT IMeshFactoryMng
{
 public:

  //! Frees the resources.
  virtual ~IMeshFactoryMng() = default;

 public:

  //! Associated mesh manager
  virtual IMeshMng* meshMng() const =0;

  /*!
   * \brief Creates a mesh or a sub-mesh.
   *
   * The created mesh is automatically added to the associated meshMng().
   */
  virtual IPrimaryMesh* createMesh(const MeshBuildInfo& build_info) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
