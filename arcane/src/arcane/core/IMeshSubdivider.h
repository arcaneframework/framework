// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshSubdivider.h                                           (C) 2000-2024 */
/*                                                                           */
/* Interface for a mesh subdivision service.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSUBDIVIDER_H
#define ARCANE_CORE_IMESHSUBDIVIDER_H
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
 * \brief IMeshSubdivider
 * \warning Experimental. Do not use outside of Arcane
 */
class ARCANE_CORE_EXPORT IMeshSubdivider
{
 public:

  virtual ~IMeshSubdivider() = default; //<! Releases resources

 public:

  //! Subdivides the mesh \a mesh
  virtual void subdivideMesh(IPrimaryMesh* mesh) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
