// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalConnectivityIndex.h                             (C) 2000-2021 */
/*                                                                           */
/* Index of a family in the connectivity accessible via ItemInternal.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMINTERNALCONNECTIVITYINDEX_H
#define ARCANE_MESH_ITEMINTERNALCONNECTIVITYINDEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshUtils.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalIndexT to access nodes
class NodeInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::NODE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalConnectivityIndexT to access edges
class EdgeInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::EDGE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalConnectivityIndexT to access faces
class FaceInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::FACE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalConnectivityIndexT to access cells
class CellInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::CELL_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalConnectivityIndexT to access HParents
class HParentInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HPARENT_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Specialization of \a CompactItemItemInternalConnectivityIndexT to access HParents
class HChildInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HCHILD_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
