// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshPatchInternal.h                               (C) 2000-2025 */
/*                                                                           */
/* Information about an AMR patch of a Cartesian mesh.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHPATCHINTERNAL_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHPATCHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshPatchInternal
{

 public:

  virtual ~ICartesianMeshPatchInternal() = default;

 public:

  virtual AMRPatchPosition& positionRef() = 0;
  virtual void setPosition(const AMRPatchPosition& position) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
