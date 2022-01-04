// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalConnectivityIndex.h                             (C) 2000-2021 */
/*                                                                           */
/* Indice d'une famille dans la connectivité accessible via ItemInternal.    */
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

//! Spécialisation de \a CompactItemItemInternalIndexT pour accéder aux noeuds
class NodeInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::NODE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemItemInternalConnectivityIndexT pour accéder aux arêtes
class EdgeInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::EDGE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemItemInternalConnectivityIndexT pour accéder aux faces
class FaceInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::FACE_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemItemInternalConnectivityIndexT pour accéder aux mailles
class CellInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::CELL_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemItemInternalConnectivityIndexT pour accéder aux HParent
class HParentInternalConnectivityIndex
{
 public:
  static Integer connectivityIndex() { return ItemInternalConnectivityList::HPARENT_IDX; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation de \a CompactItemItemInternalConnectivityIndexT pour accéder aux HParent
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
