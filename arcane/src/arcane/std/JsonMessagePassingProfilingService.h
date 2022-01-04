// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JsonMessagePassingProfilingService.h                        (C) 2000-2019 */
/*                                                                           */
/* Informations de performances du "message passing" au format JSON          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_JSONMESSAGEPASSINGPROFILINGSERVICE_H
#define ARCANE_STD_JSONMESSAGEPASSINGPROFILINGSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/AbstractService.h"
#include "arcane/ISubDomain.h"
#include "arcane/ObserverPool.h"
#include "arcane/utils/IMessagePassingProfilingService.h"
#include "arcane/utils/String.h"
#include "arccore/message_passing/Stat.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de profiling du "message passing" au format JSON.
 */
class JsonMessagePassingProfilingService
: public AbstractService
, public IMessagePassingProfilingService
{
 public:

  JsonMessagePassingProfilingService(const ServiceBuildInfo& sbi);
  virtual ~JsonMessagePassingProfilingService();

 public:

  void startProfiling() override;
  void stopProfiling() override;
  void printInfos(std::ostream& output) override;
  String implName() override;

 private:

  //! Liste des statistiques par point d'entree.
  typedef std::map<String, Arccore::MessagePassing::StatData> StatDataMap;

 private:

  ISubDomain* m_sub_domain;
  ObserverPool m_observer;
  JSONWriter* m_json_writer;
  // { entry point name, {message passing infos} }
  StatDataMap m_ep_mpstat_col;
  String m_impl_name;

private:

  void _dumpCurrentIterationInJSON();

  void _updateFromBeginEntryPointEvt();
  void _updateFromEndEntryPointEvt();
  void _updateFromBeginIterationEvt();
  void _updateFromEndIterationEvt();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
