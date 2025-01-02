// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalPartitionConstraint.h                               (C) 2000-2024 */
/*                                                                           */
/* Informations sur les contraintes pour le partitionnement.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_EXTERNALPARTITIONCONSTRAINT_H
#define ARCANE_CORE_EXTERNALPARTITIONCONSTRAINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMeshPartitionConstraint.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ExternalPartitionConstraint
: public IMeshPartitionConstraint
{
 public:

  ExternalPartitionConstraint(IMesh* mesh, ArrayView<ItemGroup> constraints)
  : m_mesh(mesh)
  , m_constraints(constraints)
  {
  }

  virtual void addLinkedCells(Int64Array& linked_cells, Int32Array& linked_owners);

 private:

  IMesh* m_mesh = nullptr;
  UniqueArray<ItemGroup> m_constraints;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
