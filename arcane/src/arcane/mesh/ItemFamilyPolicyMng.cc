// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyPolicyMng.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des politiques d'une famille d'entités.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/IItemFamilyCompactPolicy.h"

#include "arcane/mesh/ItemFamilyPolicyMng.h"
#include "arcane/mesh/ItemsExchangeInfo2.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/IndirectItemFamilySerializer.h"
#include "arcane/mesh/ItemFamilyCompactPolicy.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyPolicyMng::
~ItemFamilyPolicyMng()
{
  delete m_compact_policy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyExchanger* ItemFamilyPolicyMng::
createExchanger()
{
  ItemsExchangeInfo2* exchanger = _createExchanger();
  for( IItemFamilySerializeStepFactory* factory : m_serialize_step_factories ){
    IItemFamilySerializeStep* step = factory->createStep(m_item_family);
    if (step)
      exchanger->addSerializeStep(step);
  }
  return exchanger;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilySerializer* ItemFamilyPolicyMng::
createSerializer(bool with_flags)
{
  if (with_flags)
    throw NotSupportedException(A_FUNCINFO,"serialisation with 'with_flags==true'");
  return new IndirectItemFamilySerializer(m_item_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyPolicyMng::
addSerializeStep(IItemFamilySerializeStepFactory* factory)
{
  ARCANE_CHECK_POINTER(factory);
  m_serialize_step_factories.add(factory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyPolicyMng::
removeSerializeStep(IItemFamilySerializeStepFactory* factory)
{
  ARCANE_CHECK_POINTER(factory);
  Integer index = -1;
  for( Integer i=0, n=m_serialize_step_factories.size(); i<n; ++i ){
    if (m_serialize_step_factories[i]==factory){
      index = i;
      break;
    }
  }
  if (index==(-1))
    throw ArgumentException(A_FUNCINFO,"factory not in list");
  m_serialize_step_factories.remove(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemsExchangeInfo2* ItemFamilyPolicyMng::
_createExchanger()
{
  return new ItemsExchangeInfo2(m_item_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
