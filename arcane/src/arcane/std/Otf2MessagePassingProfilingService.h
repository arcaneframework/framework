// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2MessagePassingProfilingService.h                        (C) 2000-2025 */
/*                                                                           */
/* Informations de performances du "message passing" au format Otf2          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_OTF2MESSAGEPASSINGPROFILINGSERVICE_H
#define ARCANE_STD_INTERNAL_OTF2MESSAGEPASSINGPROFILINGSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMessagePassingProfilingService.h"
#include "arcane/utils/String.h"

#include "arcane/core/AbstractService.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"

#include "arcane/std/internal/Otf2LibWrapper.h"
#include "arcane/std/internal/Otf2MpiProfiling.h"

#include "arccore/message_passing/Stat.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace MP = ::Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de profiling du "message passing" au format JSON.
 */
class Otf2MessagePassingProfilingService
: public AbstractService
, public IMessagePassingProfilingService
{
 public:
  explicit Otf2MessagePassingProfilingService(const ServiceBuildInfo& sbi);
  ~Otf2MessagePassingProfilingService() noexcept override;

  void startProfiling() override;
  void stopProfiling() override;
  void printInfos(std::ostream& output) override;
	String implName() override;

 private:
	void _updateFromBeginEntryPointEvt();
	void _updateFromEndEntryPointEvt();
	void _updateFromSynchronizeEvt(const VariableSynchronizerEventArgs& arg);

  ISubDomain* m_sub_domain = nullptr;
  Otf2LibWrapper m_otf2_wrapper;
  Otf2MpiProfiling m_otf2_prof;
  MP::IProfiler* m_prof_backup = nullptr;
  ObserverPool m_observer;
	EventObserverPool m_observer_pool;
	String m_impl_name;
  MP::IControlDispatcher* m_control_dispatcher = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}  // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
