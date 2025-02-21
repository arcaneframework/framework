// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtils.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires associées aux 'IParallelMng'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ParallelMngUtils.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelMngUtilsFactory.h"
#include "arcane/core/internal/IParallelMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe ' friend'  de IParallelMng permettant d'accéder à
 * IParallelMng::_internalUtilsFactory() const;
 */
class ParallelMngUtilsAccessor
{
 public:

  static Ref<IGetVariablesValuesParallelOperation>
  createGetVariablesValuesOperation(IParallelMng* pm)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createGetVariablesValuesOperation(pm);
  }

  static Ref<ITransferValuesParallelOperation>
  createTransferValuesOperation(IParallelMng* pm)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createTransferValuesOperation(pm);
  }

  static Ref<IParallelExchanger>
  createExchanger(IParallelMng* pm)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createExchanger(pm);
  }

  static Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm, IItemFamily* family)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createSynchronizer(pm, family);
  }

  static Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm, const ItemGroup& group)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createSynchronizer(pm, group);
  }

  static Ref<IParallelTopology>
  createTopology(IParallelMng* pm)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createTopology(pm);
  }

  static Ref<IParallelMng>
  createSubParallelMngRef(IParallelMng* pm, Int32 color, Int32 key)
  {
    ARCANE_CHECK_POINTER(pm);
    return pm->_internalApi()->createSubParallelMngRef(color, key);
  }

  static Ref<ISerializeMessage>
  createSendSerializeMessageRef(IParallelMng* pm, Int32 rank)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createSendSerializeMessage(pm, rank);
  }

  static Ref<ISerializeMessage>
  createReceiveSerializeMessageRef(IParallelMng* pm, Int32 rank)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createReceiveSerializeMessage(pm, rank);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::ParallelMngUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IGetVariablesValuesParallelOperation>
createGetVariablesValuesOperationRef(IParallelMng* pm)
{
  return ParallelMngUtilsAccessor::createGetVariablesValuesOperation(pm);
}

Ref<ITransferValuesParallelOperation>
createTransferValuesOperationRef(IParallelMng* pm)
{
  return ParallelMngUtilsAccessor::createTransferValuesOperation(pm);
}

Ref<IParallelExchanger>
createExchangerRef(IParallelMng* pm)
{
  return ParallelMngUtilsAccessor::createExchanger(pm);
}

Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm,IItemFamily* family)
{
  return ParallelMngUtilsAccessor::createSynchronizer(pm,family);
}

Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm,const ItemGroup& group)
{
  return ParallelMngUtilsAccessor::createSynchronizer(pm,group);
}

Ref<IParallelTopology>
createTopologyRef(IParallelMng* pm)
{
  return ParallelMngUtilsAccessor::createTopology(pm);
}

Ref<IParallelMng>
createSubParallelMngRef(IParallelMng* pm, Int32 color, Int32 key)
{
  return ParallelMngUtilsAccessor::createSubParallelMngRef(pm, color, key);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::ParallelMngUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
