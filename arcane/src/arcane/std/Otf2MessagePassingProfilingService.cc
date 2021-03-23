// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2MessagePassingProfilingService.cc                       (C) 2000-2020 */
/*                                                                           */
/* Informations de performances du "message passing" au format Otf2          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/Otf2MessagePassingProfilingService.h"

#include "arcane/ServiceFactory.h"
#include "arcane/CommonVariables.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/utils/ITraceMng.h"
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

	// On initialise la lib otf2
	m_otf2_wrapper.init(m_sub_domain->application()->applicationName());

  IMessagePassingMng* mp_mng = m_sub_domain->parallelMng()->messagePassingMng();
  IControlDispatcher* cd = mp_mng->dispatchers()->controlDispatcher();
  m_control_dispatcher = cd;

	// On sauvegarde l'ancien service de profiling MPI pour le repositionner
	m_prof_backup = cd->profiler();

  // On positionne le decorateur OTF2 pour les appels MPI
	cd->setProfiler(&m_otf2_prof);

  // Lorsqu'on lance le profiling on commence l'observation des evts du timeLoopMng
  // Evt de debut de point d'entree
  m_observer.addObserver(this, &Otf2MessagePassingProfilingService::_updateFromBeginEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::BeginEntryPoint));

  // Evt de fin de point d'entree
  m_observer.addObserver(this, &Otf2MessagePassingProfilingService::_updateFromEndEntryPointEvt,
                         m_sub_domain->timeLoopMng()->observable(eTimeLoopEventType::EndEntryPoint));

  // C'est la meme fct pour tous les evts de synchronize
  auto sync_evt_handler = std::function<void(const Arcane::VariableSynchronizerEventArgs&)>(
			std::bind(&Otf2MessagePassingProfilingService::_updateFromSynchronizeEvt, this, std::placeholders::_1));

  // Evt de synchro sur les variables aux mailles
  m_sub_domain->defaultMesh()->cellFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
  		                                                                                       sync_evt_handler);
	// Evt de synchro sur les variables aux noeuds
	m_sub_domain->defaultMesh()->nodeFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// Evt de synchro sur les variables aux arretes
	m_sub_domain->defaultMesh()->edgeFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// Evt de synchro sur les variables aux faces
	m_sub_domain->defaultMesh()->faceFamily()->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,
																																														 sync_evt_handler);
	// TODO: le faire pour les variables materiaux

  // On commence le profiling du programme
  OTF2_EvtWriter_ProgramBegin(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
  		                        m_otf2_wrapper.getApplicationNameId(), 0, NULL);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
stopProfiling()
{
	// On enleve le decorateur OTF2 des appels MPI
	if (m_control_dispatcher)
    m_control_dispatcher->setProfiler(m_prof_backup);

	// On termine le profiling du programme
	OTF2_EvtWriter_ProgramEnd(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(), 0);
	m_otf2_wrapper.finalize();

	// Arret des observations
  m_observer.detachAll();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void Otf2MessagePassingProfilingService::
printInfos(std::ostream& output)
{
  ARCANE_UNUSED(output);
	// TODO: fait on qqch ici ?
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
	// Recuperation du nom du pt d'entree
	const String& ep_name(m_sub_domain->timeLoopMng()->currentEntryPoint()->fullName());

	OTF2_EvtWriter_Enter(m_otf2_wrapper.getEventWriter(), NULL, Otf2LibWrapper::getTime(),
			                 m_otf2_wrapper.getEntryPointId(ep_name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Otf2MessagePassingProfilingService::
_updateFromEndEntryPointEvt()
{
	// Recuperation du nom du pt d'entree
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
