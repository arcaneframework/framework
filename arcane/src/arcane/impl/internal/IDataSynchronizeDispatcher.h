﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeDispatcher.h                                 (C) 2000-2023 */
/*                                                                           */
/* Gestion de la synchronisation d'une instance de 'IData'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/impl/IDataSynchronizeImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DataSynchronizeResult;
class DataSynchronizeMemory;
class IVariableSynchronizerDispatcher;
class INumericDataInternal;
class IBufferCopier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un DataSynchronizeDispatcher.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeDispatcherBuildInfo
{
 public:

  DataSynchronizeDispatcherBuildInfo(IParallelMng* pm,
                                     Ref<IDataSynchronizeImplementation> sync_impl,
                                     Ref<DataSynchronizeInfo> sync_info,
                                     Ref<DataSynchronizeMemory> memory,
                                     Ref<IBufferCopier> copier)
  : m_parallel_mng(pm)
  , m_synchronize_implementation(sync_impl)
  , m_synchronize_info(sync_info)
  , m_synchronize_memory(memory)
  , m_buffer_copier(copier)
  {}

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }
  Ref<IDataSynchronizeImplementation> synchronizeImplementation() const { return m_synchronize_implementation; }
  Ref<DataSynchronizeInfo> synchronizeInfo() const { return m_synchronize_info; }
  Ref<DataSynchronizeMemory> synchronizeMemory() const { return m_synchronize_memory; }
  Ref<IBufferCopier> bufferCopier() const { return m_buffer_copier; }

 private:

  IParallelMng* m_parallel_mng = nullptr;
  Ref<IDataSynchronizeImplementation> m_synchronize_implementation;
  Ref<DataSynchronizeInfo> m_synchronize_info;
  Ref<DataSynchronizeMemory> m_synchronize_memory;
  Ref<IBufferCopier> m_buffer_copier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface pour gérer la synchronisation d'une donnée.
 *
 * Il faut utiliser create() pour créer une implémentation pour cette
 * interface.
 *
 * Il faut appeler compute() avant de pouvoir utiliser l'instance et aussi
 * lorsque la famille d'entité associée évolue.
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeDispatcher
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IDataSynchronizeDispatcher() = default;

 public:

  /*!
   * \brief Recalcule les informations nécessaires après une mise à jour des informations
   * de \a DataSynchronizeInfo.
   */
  virtual void compute() = 0;

  /*!
   * \brief Commence l'exécution pour la synchronisation pour la donnée \a data.
   */
  virtual void beginSynchronize(INumericDataInternal* data, bool is_compare_sync) = 0;

  /*!
   * \brief Termine la synchronisation.
   *
   * Il faut avoir appelé beginSynchronize() avant.
   */
  virtual DataSynchronizeResult endSynchronize() = 0;

 public:

  static Ref<IDataSynchronizeDispatcher>
  create(const DataSynchronizeDispatcherBuildInfo& build_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la synchronisation d'une liste de variables.
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeMultiDispatcher
{
 public:

  virtual ~IDataSynchronizeMultiDispatcher() = default;

 public:

  /*!
   * \brief Recalcule les informations nécessaires après une mise à jour des informations
   * de \a DataSynchronizeInfo.
   */
  virtual void compute() = 0;
  virtual void synchronize(ConstArrayView<IVariable*> vars) = 0;

 public:

  static IDataSynchronizeMultiDispatcher* create(const DataSynchronizeDispatcherBuildInfo& bi);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
