// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PapiPerformanceService.h                                    (C) 2000-2023 */
/*                                                                           */
/* Informations de performances utilisant PAPI.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_PAPIPERFORMANCESERVICE_H
#define ARCANE_STD_PAPIPERFORMANCESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IProfilingService.h"

#include "arcane/AbstractService.h"

#include <papi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de profiling utilisant la bibliothèque PAPI.
 */
class PapiPerformanceService
: public AbstractService
, public IProfilingService
{
 public:

  explicit PapiPerformanceService(const ServiceBuildInfo& sbi);
  ~PapiPerformanceService() override;

 public:

  void initialize() override;
  bool isInitialized() const override { return m_is_initialized; }
  void startProfiling() override;
  void switchEvent() override;
  void stopProfiling() override;
  void printInfos(bool dump_file) override;
  void getInfos(Int64Array&) override;
  void dumpJSON(JSONWriter& writer) override;
  void reset() override;
  ITimerMng* timerMng() override;

 private:

  int m_period = 0;
  int m_event_set = 0;
  bool m_only_flops = false;
  bool m_is_running = false;
  bool m_is_initialized = false;
  ITimerMng* m_timer_mng = nullptr;
  IApplication* m_application = nullptr;

 private:

  void _printFlops();
  bool _addEvent(int event_code,int event_index);
  static void arcane_papi_handler(int EventSet, void *address,
                                  long_long overflow_vector, void *context);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
