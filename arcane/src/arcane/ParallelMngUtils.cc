// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtils.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires associées aux 'IParallelMng'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ParallelMngUtils.h"

#include "arcane/IParallelMng.h"
#include "arcane/IParallelMngUtilsFactory.h"

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
  createSynchronizer(IParallelMng* pm,IItemFamily* family)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createSynchronizer(pm,family);
  }

  static Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm,const ItemGroup& group)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createSynchronizer(pm,group);
  }

  static Ref<IParallelTopology>
  createTopology(IParallelMng* pm)
  {
    ARCANE_CHECK_POINTER(pm);
    auto f = pm->_internalUtilsFactory();
    return f->createTopology(pm);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::ParallelMngUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
