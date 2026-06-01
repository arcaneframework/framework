// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2MessagePassingProfilingService.cc                       (C) 2000-2020 */
/*                                                                           */
/* Performance information for "message passing" in Otf2 format              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/Otf2MessagePassingProfilingService.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IEntryPoint.h"

#include "arcane/parallel/IStat.h"

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IDispatchers.h"
#include "arccore/message_passing/IControlDispatcher.h"

#include <algorithm>
#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(Otf2MessagePassingProfilingService,
                        ServiceProperty("Otf2MessagePassingProfiling",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMessagePassingProfilingService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Otf2MessagePassingProfilingService::
Otf2MessagePassingProfilingService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
, m_otf2_wrapper(sbi.subDomain())
, m_otf2_prof(&m_otf2_wrapper)
, m_prof_backup(nullptr)
, m_impl_name("Otf2MessagePassingProfiling")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Otf2MessagePassingProfilingService::
~Otf2MessagePassingProfilingService() noexcept
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
startProfiling()
{
	// DBG
	//info() << "========== OTF2 PROFILING SERVICE ! ==========";

	// Initialize the otf2 library
	m_otf2_wrapper.init(m_sub_domain->application()->applicationName());

  IMessagePassingMng* mp_mng = m_sub_domain->parallelMng()->messagePassingMng();
  IControlDispatcher* cd = mp_mng->dispatchers()->controlDispatcher();
  m_control_dispatcher = cd;

	// Save the old MPI profiling service to reposition it
	m_prof_backup = cd->profiler();

  // Position the OTF2 decorator for MPI calls
	cd->setProfiler(&m_otf2_prof);

  // When starting profiling, we begin observing timeLoopMng events
  // Entry point start event
  m_observer.addObserver(this, &Otf2MessagePassingProfilingService::_updateFromBeginEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::BeginEntryPoint));

  // Entry point end event
  m_observer.addObserver(this, &Otf2MessagePassingProfilingService::_updateFromEndEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::EndEntryPoint));

  // It is the same function for all synchronization events
  auto sync_evt_handler = std::function<void(const Arcane::VariableSynchronizerEventArgs&)>(
			std::bind(&Otf2MessagePassingProfilingService::_updateFromSynchronizeEvt, this, std::placeholders::_1));

  // Synchronization event on cell variables
  m_sub_domain->defaultMesh()->cellFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
  		                                                                                       sync_evt_handler);
	// Synchronization event on node variables
	m_sub_domain->defaultMesh()->nodeFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// Synchronization event on edge variables
	m_sub_domain->defaultMesh()->edgeFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// Synchronization event on face variables
	m_sub_domain->defaultMesh()->faceFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// TODO: do it for material variables

  // Start program profiling
  OTF2_EvtWriter_ProgramBegin(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
  		                        m_otf2_wrapper.getApplicationNameId(), 0, NULL);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
stopProfiling()
{
	// Remove the OTF2 decorator from MPI calls
	if (m_control_dispatcher)
    m_control_dispatcher->setProfiler(m_prof_backup);

	// Finish program profiling
	OTF2_EvtWriter_ProgramEnd(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(), 0);
	m_otf2_wrapper.finalize();

	// Stop observations
  m_observer.detachAll();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void Otf2MessagePassingProfilingService::
printInfos(std::ostream& output)
{
  ARCANE_UNUSED(output);
	// TODO: should we do something here?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
String Otf2MessagePassingProfilingService::
implName()
{
	return m_impl_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
_updateFromBeginEntryPointEvt()
{
	// Retrieve the entry point name
	const String& ep_name(m_sub_domain->timeLoopMng()->currentEntryPoint()->fullName());

	OTF2_EvtWriter_Enter(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
			                 m_otf2_wrapper.getEntryPointId(ep_name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
_updateFromEndEntryPointEvt()
{
	// Retrieve the entry point name
	const String& ep_name(m_sub_domain->timeLoopMng()->currentEntryPoint()->fullName());

	OTF2_EvtWriter_Leave(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
			                 m_otf2_wrapper.getEntryPointId(ep_name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
_updateFromSynchronizeEvt(const VariableSynchronizerEventArgs& arg)
{
	if (arg.state() == VariableSynchronizerEventArgs::State::BeginSynchronize)
		OTF2_EvtWriter_Enter(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
		                     m_otf2_wrapper.getSynchronizeId());
	else
		OTF2_EvtWriter_Leave(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
		                     m_otf2_wrapper.getSynchronizeId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}  // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
