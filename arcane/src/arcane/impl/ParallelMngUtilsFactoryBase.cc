// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtilsFactoryBase.cc                              (C) 2000-2021 */
/*                                                                           */
/* Classe de base d'une fabrique pour les fonctions utilitaires de           */
/* IParallelMng.                                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ParallelMngUtilsFactoryBase.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/impl/GetVariablesValuesParallelOperation.h"
#include "arcane/impl/TransferValuesParallelOperation.h"
#include "arcane/impl/ParallelExchanger.h"
#include "arcane/impl/ParallelTopology.h"
#include "arcane/impl/VariableSynchronizer.h"

#include "arcane/DataTypeDispatchingDataVisitor.h"

#include "arcane/IItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IGetVariablesValuesParallelOperation> ParallelMngUtilsFactoryBase::
createGetVariablesValuesOperation(IParallelMng* pm)
{
  return makeRef<IGetVariablesValuesParallelOperation>(new GetVariablesValuesParallelOperation(pm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ITransferValuesParallelOperation> ParallelMngUtilsFactoryBase::
createTransferValuesOperation(IParallelMng* pm)
{
  return makeRef<ITransferValuesParallelOperation>(new TransferValuesParallelOperation(pm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelExchanger> ParallelMngUtilsFactoryBase::
createExchanger(IParallelMng* pm)
{
  return makeRef<IParallelExchanger>(new ParallelExchanger(pm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelTopology> ParallelMngUtilsFactoryBase::
createTopology(IParallelMng* pm)
{
  ParallelTopology* t = new ParallelTopology(pm);
  t->initialize();
  return makeRef<IParallelTopology>(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IVariableSynchronizer> ParallelMngUtilsFactoryBase::
createSynchronizer(IParallelMng* pm,IItemFamily* family)
{
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
  VariableSynchronizeDispatcherBuildInfo bi(pm,nullptr);
  auto vd = new VariableSynchronizerDispatcher(pm,DispatcherType::create<SimpleVariableSynchronizeDispatcher>(bi));
  return makeRef<IVariableSynchronizer>(new VariableSynchronizer(pm,family->allItems(),vd));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IVariableSynchronizer> ParallelMngUtilsFactoryBase::
createSynchronizer(IParallelMng* pm,const ItemGroup& group)
{
  SharedPtrT<GroupIndexTable> table = group.localIdToIndex();
  VariableSynchronizeDispatcherBuildInfo bi(pm,table.get());
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
  auto vd = new VariableSynchronizerDispatcher(pm,DispatcherType::create<SimpleVariableSynchronizeDispatcher>(bi));
  return makeRef<IVariableSynchronizer>(new VariableSynchronizer(pm,group,vd));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
