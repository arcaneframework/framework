// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFFamilyPolicyMng.cc                                       (C) 2000-2016 */
/*                                                                           */
/* Gestionnaire des politiques d'une famille de DoF.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"
#include "arcane/mesh/IndirectItemFamilySerializer.h"
#include "arcane/mesh/DoFFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DoFFamilyCompactPolicy
: public ItemFamilyCompactPolicy
{
 public:
  DoFFamilyCompactPolicy(ItemFamily* family) : ItemFamilyCompactPolicy(family){}
 public:
  void updateInternalReferences(IMeshCompacter* compacter) override
  {
    // Pour l'instant ne fait rien car c'est la famille source qui gère la
    // mise à jour dans ItemFamily::beginCompactItems().
    ARCANE_UNUSED(compacter);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des politiques d'une famille de DoF.
 */
class ARCANE_MESH_EXPORT DoFFamilyPolicyMng
: public ItemFamilyPolicyMng
{
 public:
  DoFFamilyPolicyMng(DoFFamily* family)
  : ItemFamilyPolicyMng(family,new DoFFamilyCompactPolicy(family))
  , m_family(family){}
 public:
  IItemFamilySerializer* createSerializer(bool use_flags) override
  {
    if (use_flags)
      throw NotSupportedException(A_FUNCINFO,"serialisation with 'use_flags==true'");
    return new IndirectItemFamilySerializer(m_family);
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

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
