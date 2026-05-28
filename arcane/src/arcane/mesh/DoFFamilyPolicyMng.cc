// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFFamilyPolicyMng.cc                                       (C) 2000-2016 */
/*                                                                           */
/* Manager of policies for a DoF family.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/ItemFamilySerializer.h"
#include "arcane/mesh/IndirectItemFamilySerializer.h"
#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DoFFamilyCompactPolicy
: public ItemFamilyCompactPolicy
{
 public:

  DoFFamilyCompactPolicy(ItemFamily* family)
  : ItemFamilyCompactPolicy(family)
  {}

 public:

  void updateInternalReferences(IMeshCompacter* compacter) override
  {
    // Does nothing for now because the source family handles the
    // update in ItemFamily::beginCompactItems().
    ARCANE_UNUSED(compacter);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Manager of policies for a DoF family.
 */
class ARCANE_MESH_EXPORT DoFFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:

  DoFFamilyPolicyMng(DoFFamily* family)
  : ItemFamilyPolicyMng(family, new DoFFamilyCompactPolicy(family))
  , m_family(family)
  {}

 public:

  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO, "serialisation with 'use_flags==true'");

    IMesh* mesh = m_family->mesh();
    DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
    return new ItemFamilySerializer(m_family, m_family, dmesh->incrementalBuilder());
    //return new IndirectItemFamilySerializer(m_family);
  }

 private:

  DoFFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createDoFFamilyPolicyMng(ItemFamily* family)
{
  DoFFamily* f = ARCANE_CHECK_POINTER(dynamic_cast<DoFFamily*>(family));
  return new DoFFamilyPolicyMng(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
