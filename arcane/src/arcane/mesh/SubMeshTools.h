// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubMeshTools.h                                              (C) 2000-2024 */
/*                                                                           */
/* Algorithmes spécifiques aux sous-maillages.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_SUBMESHTOOLS_H
#define ARCANE_MESH_SUBMESHTOOLS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette section devrait contenir à terme l'ensemble de l'implémentation spécifique des sous-maillages
 */

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

class IItemFamily;
class IParallelMng;
class IMesh;
class ItemInternal;
class Cell;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

class DynamicMesh;
class DynamicMeshIncrementalBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SubMeshTools
: public TraceAccessor
{
 public:
  SubMeshTools(DynamicMesh * mesh, DynamicMeshIncrementalBuilder * mesh_builder);
  ~SubMeshTools();

 public:
  void removeDeadGhostCells();
  void removeGhostMesh();
  void removeFloatingItems();
  void updateGhostMesh();
  void updateGhostFamily(IItemFamily * family);
  static void display(IMesh * mesh, const String msg = String());

 private:

  void _fillGhostItems(ItemFamily* family, Array<Int32>& items_local_id);
  void _fillFloatingItems(ItemFamily* family, Array<Int32>& items_local_id);
  void _checkValidItemOwner();
  void _checkFloatingItems();

 private:
  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;
  IParallelMng* m_parallel_mng;

 private:

  void _updateGroups();
  void _removeCell(Cell cell);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
