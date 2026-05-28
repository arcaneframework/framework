// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionConstraintMng.h                                (C) 2000-2009 */
/*                                                                           */
/* Mesh partitioning constraint manager.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHPARTITIONCONSTRAINTMNG_H
#define ARCANE_MESH_MESHPARTITIONCONSTRAINTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/List.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/core/IMeshPartitionConstraintMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh partitioning constraint manager.
 */
class MeshPartitionConstraintMng
: public TraceAccessor
, public IMeshPartitionConstraintMng
{
 private:

  class Helper;

 public:

  MeshPartitionConstraintMng(IMesh* mesh);
  virtual ~MeshPartitionConstraintMng();

 public:

  virtual void addConstraint(IMeshPartitionConstraint* constraint);
  virtual void removeConstraint(IMeshPartitionConstraint* constraint);
  virtual void computeAndApplyConstraints();
  virtual void computeConstraintList(Int64MultiArray2& tied_uids);

  virtual void addWeakConstraint(IMeshPartitionConstraint* constraint);
  virtual void removeWeakConstraint(IMeshPartitionConstraint* constraint);
  virtual void computeAndApplyWeakConstraints();
  virtual void computeWeakConstraintList(Int64MultiArray2& tied_uids);

 private:

  IMesh* m_mesh;
  bool m_is_debug;
  List<IMeshPartitionConstraint*> m_constraints;
  List<IMeshPartitionConstraint*> m_weak_constraints;

 private:

  void _computeAndApplyConstraints(Helper& h);
  void _computeAndApplyWeakConstraints(Helper& h);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/

#endif
