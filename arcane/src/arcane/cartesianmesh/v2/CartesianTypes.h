// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianTypes.h                                            (C) 2000-2021 */
/*                                                                           */
/* Types for Cartesian meshes.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANTYPES_H
#define ARCANE_CARTESIANMESH_CARTESIANTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::CartesianMesh::V2
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Type for Cartesian triplets (i,j,k) and dimension triplets (ni,nj,nk)
 */
using LocalIdType3 = Arcane::LocalIdType[3];

using UniqueIdType3 = Arcane::UniqueIdType[3];

using LocalIdType4 = Arcane::LocalIdType[4];

using IdxType = std::array<Arcane::Int64, 3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Side type (previous or next) for a given direction
 */
enum eMeshSide
{
  //! Previous side
  MS_previous = 0,
  //! Next side
  MS_next = 1,
  //! Max number of valid sides
  MS_max = 2,
  //! Invalid or uninitialized side
  MS_invalid = (-1)
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CartesianMesh::V2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
