﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtilsFactoryBase.h                               (C) 2000-2021 */
/*                                                                           */
/* Classe de base d'une fabrique pour les fonctions utilitaires de           */
/* IParallelMng.                                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_PARALLELMNGUTILSFACTORY_H
#define ARCANE_IMPL_PARALLELMNGUTILSFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IParallelMngUtilsFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une fabrique pour les fonctions utilitaires de IParallelMng.
 */
class ARCANE_IMPL_EXPORT ParallelMngUtilsFactoryBase
: public IParallelMngUtilsFactory
{
 public:

  Ref<IGetVariablesValuesParallelOperation> createGetVariablesValuesOperation(IParallelMng* pm) override;
  Ref<ITransferValuesParallelOperation> createTransferValuesOperation(IParallelMng* pm) override;
  Ref<IParallelExchanger> createExchanger(IParallelMng* pm) override;
  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,IItemFamily* family) override;
  Ref<IVariableSynchronizer> createSynchronizer(IParallelMng* pm,const ItemGroup& group) override;
  Ref<IParallelTopology> createTopology(IParallelMng* pm) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
