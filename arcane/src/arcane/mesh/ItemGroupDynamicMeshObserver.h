// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*                                                                           */
/* Outil d'observation du maillage                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMGROUPDYNAMICMESHOBSERVER_H
#define ARCANE_MESH_ITEMGROUPDYNAMICMESHOBSERVER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/ItemGroupObserver.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemGroupDynamicMeshObserver
: public IItemGroupObserver
{
 public:
  ItemGroupDynamicMeshObserver(DynamicMesh * mesh) : m_mesh(mesh) { }
  virtual ~ItemGroupDynamicMeshObserver() { }

  void executeExtend(const Int32ConstArrayView * new_items_info);

  void executeReduce(const Int32ConstArrayView * info);

  void executeCompact(const Int32ConstArrayView * pinfo);

  void executeInvalidate();

  bool needInfo() const { return true; }

private:
  DynamicMesh * m_mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_ITEMGROUPDYNAMICMESHOBSERVER_H */
