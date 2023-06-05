// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataSynchronizeImplementation.h                            (C) 2000-2023 */
/*                                                                           */
/* Interface pour l'implémentation d'une synchronisation de variables.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_IDATASYNCHRONIZEIMPLEMENTATION_H
#define ARCANE_IMPL_IDATASYNCHRONIZEIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataSynchronizeBuffer;
class IParallelMng;
class DataSynchronizeInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IDataSynchronizeImplementation;
class IDataSynchronizeImplementationFactory;
class AbstractDataSynchronizeImplementation;

using IGenericVariableSynchronizerDispatcher
ARCANE_DEPRECATED_REASON("Use 'IDataSynchronizeImplementation' instead") = IDataSynchronizeImplementation;

using IGenericVariableSynchronizerDispatcherFactory
ARCANE_DEPRECATED_REASON("Use 'IDataSynchronizeImplementationFactory' instead") = IDataSynchronizeImplementationFactory;

using AbstractGenericVariableSynchronizerDispatcher
ARCANE_DEPRECATED_REASON("Use 'AbstractDataSynchronizeImplementation' instead") = AbstractDataSynchronizeImplementation;

using ItemGroupSynchronizeInfo
ARCANE_DEPRECATED_REASON("Use 'DataSynchronizeInfo' instead") = DataSynchronizeInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un dispatcher générique.
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeImplementation
{
 public:

  virtual ~IDataSynchronizeImplementation() = default;

 public:

  virtual void setDataSynchronizeInfo(DataSynchronizeInfo* sync_info) = 0;
  virtual void compute() = 0;
  virtual void beginSynchronize(IDataSynchronizeBuffer* buf) = 0;
  virtual void endSynchronize(IDataSynchronizeBuffer* buf) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fabrique dispatcher générique.
 */
class ARCANE_IMPL_EXPORT IDataSynchronizeImplementationFactory
{
 public:

  virtual ~IDataSynchronizeImplementationFactory() = default;

 public:

  virtual Ref<IDataSynchronizeImplementation> createInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Classe de base abstraite pour les implémentations génériques.
 *
 * Elle permet de conserver les informations sur le groupe à synchroniser.
 */
class AbstractDataSynchronizeImplementation
: public IDataSynchronizeImplementation
{
 public:

  void setDataSynchronizeInfo(DataSynchronizeInfo* sync_info) final { m_sync_info = sync_info; }

 protected:

  DataSynchronizeInfo* _syncInfo() const { return m_sync_info; }

 private:

  DataSynchronizeInfo* m_sync_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
