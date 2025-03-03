// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PatchAMR.h                                                  (C) 2000-2025 */
/*                                                                           */
/* TODO                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_PATCHAMRPOSITION_H
#define ARCANE_CARTESIANMESH_PATCHAMRPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ICartesianMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT PatchAMRPosition
{
 public:

  PatchAMRPosition(const Real3& position, const Real3& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(true)
  {}

  PatchAMRPosition(const Real2& position, const Real2& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(false)
  {}

 public:

  Real3 position() const
  {
    return m_position;
  }

  Real3 length() const
  {
    return m_length;
  }

  bool is3d() const
  {
    return m_is_3d;
  }

  void cellsInPatch(ICartesianMesh* cmesh, UniqueArray<Int32>& cells_local_id) const;

 private:
  Real3 m_position;
  Real3 m_length;
  bool m_is_3d;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

