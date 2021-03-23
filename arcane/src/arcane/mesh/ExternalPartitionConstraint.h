// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalPartitionConstraint.cc                              (C) 2000-2014 */
/*                                                                           */
/* Informations sur les contraintes pour le partitionnement.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EXTERNALPARTITIONCONSTRAINT_H
#define ARCANE_MESH_EXTERNALPARTITIONCONSTRAINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

#include "arcane/VariableTypes.h"

#include "arcane/IMeshPartitionConstraint.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ExternalPartitionConstraint
: public IMeshPartitionConstraint
{
 public:
  ExternalPartitionConstraint(IMesh* mesh, ArrayView<ItemGroup> constraints)
    : m_mesh(mesh), m_constraints(constraints)
  {
  }

  virtual void addLinkedCells(Int64Array& linked_cells,Int32Array& linked_owners);

private:
  IMesh* m_mesh;
  UniqueArray<ItemGroup> m_constraints;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
