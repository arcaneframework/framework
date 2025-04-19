// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshKind.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Caractéristiques d'un maillage.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshKind.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  const char* _toName(eMeshStructure r)
  {
    switch (r) {
    case eMeshStructure::Unknown:
      return "Unknown";
    case eMeshStructure::Unstructured:
      return "Unstructured";
    case eMeshStructure::Cartesian:
      return "Cartesian";
    default:
      return "Invalid";
    }
  }
  const char* _toName(eMeshAMRKind r)
  {
    switch (r) {
    case eMeshAMRKind::None:
      return "None";
    case eMeshAMRKind::Cell:
      return "Cell";
    case eMeshAMRKind::Patch:
      return "Patch";
    case eMeshAMRKind::PatchCartesianMeshOnly:
      return "PatchCartesianMeshOnly";
    default:
      return "Invalid";
    }
  }
  const char* _toName(eMeshCellDimensionKind r)
  {
    switch (r) {
    case eMeshCellDimensionKind::MonoDimension:
      return "MonoDimension";
    case eMeshCellDimensionKind::MultiDimension:
      return "MultiDimension";
    case eMeshCellDimensionKind::NonManifold:
      return "NonManifold";
    default:
      return "Invalid";
    }
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o, eMeshStructure r)
{
  o << _toName(r);
  return o;
}

std::ostream&
operator<<(std::ostream& o, eMeshAMRKind r)
{
  o << _toName(r);
  return o;
}

std::ostream&
operator<<(std::ostream& o, eMeshCellDimensionKind r)
{
  o << _toName(r);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
