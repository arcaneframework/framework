// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.h                            (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/impl/IGenericVariableSynchronizerDispatcher.h"
#include "arcane/impl/DataSynchronizeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerDispatcher;
class VariableSynchronizerMultiDispatcher;
class IVariableSynchronizerDispatcher;
class GroupIndexTable;
class INumericDataInternal;
using IVariableSynchronizeDispatcher = IVariableSynchronizerDispatcher;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un VariableSynchronizeDispatcher.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcherBuildInfo
{
 public:

  VariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table,
                                         Ref<IDataSynchronizeImplementationFactory> factory)
  : m_parallel_mng(pm)
  , m_table(table)
  , m_factory(factory)
  {}

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  Ref<IDataSynchronizeImplementationFactory> factory() const
  {
    return m_factory;
  }

 private:

  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Ref<IDataSynchronizeImplementationFactory> m_factory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer l'envoi de la synchronisation.
 *
 * Il faut utiliser create() pour créer une implémentation pour cette
 * interface.
 *
 * Il faut appeler \a setDataSynchronizeInfo() pour initialiser
 * l'instance.
 */
class ARCANE_IMPL_EXPORT IVariableSynchronizerDispatcher
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IVariableSynchronizerDispatcher() = default;

 public:

  virtual void setItemGroupSynchronizeInfo(DataSynchronizeInfo* sync_info) = 0;

  /*!
   * \brief Recalcule les informations nécessaires après une mise à jour des informations
   * de \a DataSynchronizeInfo.
   */
  virtual void compute() = 0;

  //! Commence l'exécution pour la synchronisation pour la donnée \a data.
  virtual void beginSynchronize(INumericDataInternal* data) = 0;

  //! Termine la synchronisation.
  virtual void endSynchronize() = 0;

 public:

  static Ref<IVariableSynchronizeDispatcher>
  create(const VariableSynchronizeDispatcherBuildInfo& build_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la synchronisation d'une liste de variables.
 */
class ARCANE_IMPL_EXPORT IVariableSynchronizerMultiDispatcher
{
 public:

  virtual ~IVariableSynchronizerMultiDispatcher() = default;

 public:

  virtual void synchronize(VariableCollection vars, DataSynchronizeInfo* sync_info) = 0;

 public:

  static IVariableSynchronizerMultiDispatcher* create(const VariableSynchronizeDispatcherBuildInfo& bi);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
