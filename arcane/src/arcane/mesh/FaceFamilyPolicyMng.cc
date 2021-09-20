// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceFamilyPolicyMng.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire des politiques d'une famille de faces.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamilyNetwork.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/IndirectItemFamilySerializer.h"
#include "arcane/mesh/TiedInterfaceExchanger.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/FaceFamily.h"
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
 * \brief Gestionnaire des politiques d'une famille de faces.
 */
class ARCANE_MESH_EXPORT FaceFamilyPolicyMng
: public ItemFamilyPolicyMng
{
  class TiedInterfaceSerializeStepFactory
  : public IItemFamilySerializeStepFactory
  {
    IItemFamilySerializeStep* createStep(IItemFamily* family)
    {
      IMesh* mesh = family->mesh();
      if (mesh->hasTiedInterface()){
        DynamicMesh* x = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
        return new TiedInterfaceExchanger(x);
      }
      return nullptr;
    }
  };
 public:
  FaceFamilyPolicyMng(FaceFamily* family)
  : ItemFamilyPolicyMng(family,new StandardItemFamilyCompactPolicy(family))
  , m_family(family)
  {
    addSerializeStep(&m_tied_interface_serialize_factory);
  }
  ~FaceFamilyPolicyMng()
  {
    removeSerializeStep(&m_tied_interface_serialize_factory);
  }
 public:
  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO,"serialisation with 'use_flags==true'");
    IMesh* mesh = m_family->mesh();
    DynamicMesh* dmesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
    if(m_family->mesh()->useMeshItemFamilyDependencies())
      return new ItemFamilySerializer(m_family, m_family,dmesh->incrementalBuilder());
    else
      return new IndirectItemFamilySerializer(m_family);
  }
 private:
  FaceFamily* m_family;
  TiedInterfaceSerializeStepFactory m_tied_interface_serialize_factory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createFaceFamilyPolicyMng(ItemFamily* family)
{
  FaceFamily* f = ARCANE_CHECK_POINTER(dynamic_cast<FaceFamily*>(family));
  return new FaceFamilyPolicyMng(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
