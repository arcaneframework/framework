// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeFamilyPolicyMng.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Node family policy manager.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamilyNetwork.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/IndirectItemFamilySerializer.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/ItemFamilySerializer.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Node family policy manager.
 */
class ARCANE_MESH_EXPORT NodeFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:

  NodeFamilyPolicyMng(NodeFamily* family)
  : ItemFamilyPolicyMng(family, new StandardItemFamilyCompactPolicy(family))
  , m_family(family)
  {}

 public:

  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO, "serialisation with 'use_flags==true'");
    IMesh* mesh = m_family->mesh();
    DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
    if (m_family->mesh()->useMeshItemFamilyDependencies())
      return new ItemFamilySerializer(m_family, m_family, dmesh->incrementalBuilder());
    else
      return new IndirectItemFamilySerializer(m_family);
  }

 private:

  NodeFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createNodeFamilyPolicyMng(ItemFamily* family)
{
  NodeFamily* f = ARCANE_CHECK_POINTER(dynamic_cast<NodeFamily*>(family));
  return new NodeFamilyPolicyMng(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
