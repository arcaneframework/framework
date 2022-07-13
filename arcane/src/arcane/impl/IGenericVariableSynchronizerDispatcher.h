// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGenericVariableSynchronizerDispatcher.h                    (C) 2000-2022 */
/*                                                                           */
/* Interface d'un buffer générique pour la synchronisation de variables.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IGENERICVARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_IGENERICVARIABLESYNCHRONIZERDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataSynchronizeBuffer;
class ItemGroupSynchronizeInfo;
class IParallelMng;
class GroupIndexTable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un dispatcher générique.
 */
class ARCANE_IMPL_EXPORT IGenericVariableSynchronizerDispatcher
{
 public:

  virtual ~IGenericVariableSynchronizerDispatcher() = default;

 public:

  virtual void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) = 0;
  virtual void compute() = 0;
  virtual void beginSynchronize(IDataSynchronizeBuffer* buf) = 0;
  virtual void endSynchronize(IDataSynchronizeBuffer* buf) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fabrique dispatcher générique.
 */
class ARCANE_IMPL_EXPORT IGenericVariableSynchronizerDispatcherFactory
{
 public:

  virtual ~IGenericVariableSynchronizerDispatcherFactory() = default;

 public:

  virtual Ref<IGenericVariableSynchronizerDispatcher> createInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour construire un dispatcher générique.
 */
class ARCANE_IMPL_EXPORT GenericVariableSynchronizeDispatcherBuildInfo
{
 public:

  GenericVariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table,
                                                Ref<IGenericVariableSynchronizerDispatcherFactory> factory)
  : m_parallel_mng(pm)
  , m_table(table)
  , m_factory(factory)
  {}

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  Ref<IGenericVariableSynchronizerDispatcherFactory> factory() { return m_factory; }

 private:

  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Ref<IGenericVariableSynchronizerDispatcherFactory> m_factory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Classe de base abstraite pour les implémentations génériques.
 *
 * Elle permet de conserver les informations sur le groupe à synchroniser.
 */
class AbstractGenericVariableSynchronizerDispatcher
: public IGenericVariableSynchronizerDispatcher
{
 public:

  void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) final { m_sync_info = sync_info; }

 protected:

  ItemGroupSynchronizeInfo* _syncInfo() const { return m_sync_info; }

 private:

  ItemGroupSynchronizeInfo* m_sync_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
