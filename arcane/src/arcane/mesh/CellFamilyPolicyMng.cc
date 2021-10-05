// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamilyPolicyMng.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire des politiques d'une famille de mailles.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemFamilyNetwork.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/CellFamilySerializer.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/ItemFamilySerializer.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des politiques d'une famille de mailles.
 */
class ARCANE_MESH_EXPORT CellFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:
  CellFamilyPolicyMng(CellFamily* family)
  : ItemFamilyPolicyMng(family,new StandardItemFamilyCompactPolicy(family))
  , m_family(family){}
 public:
  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    IMesh* mesh = m_family->mesh();
    DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
    // Todo use unique_ptr ?
    if(mesh->useMeshItemFamilyDependencies())
      return new ItemFamilySerializer(m_family, m_family, dmesh->incrementalBuilder());
    else
      return new CellFamilySerializer(m_family,use_flags,dmesh->incrementalBuilder());

  }
 private:
  CellFamily* m_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createCellFamilyPolicyMng(ItemFamily* family)
{
  CellFamily* f = ARCANE_CHECK_POINTER(dynamic_cast<CellFamily*>(family));
  return new CellFamilyPolicyMng(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
