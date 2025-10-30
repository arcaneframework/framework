// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMng.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Implémentation de 'IAcceleratorMng'                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arccore/trace/TraceAccessor.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Ref.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CHECK_HAS_INIT()                        \
  if (!m_has_init)\
    ARCANE_FATAL("Invalid call because IAcceleratorMng::initialized() has not been called")

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des modules.
 */
class AcceleratorMng
: public TraceAccessor
, public IAcceleratorMng
{
 public:

  explicit AcceleratorMng(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void initialize(const AcceleratorRuntimeInitialisationInfo& runtime_info) override;
  bool isInitialized() const override { return m_has_init; }
  Runner* defaultRunner() override
  {
    CHECK_HAS_INIT();
    return m_default_runner_ref.get();
  }
  RunQueue* defaultQueue() override
  {
    CHECK_HAS_INIT();
    return m_default_queue_ref.get();
  }
  Runner runner() override
  {
    return m_default_runner;
  }
  RunQueue queue() override { return m_default_queue; }

 private:

  std::unique_ptr<Runner> m_default_runner_ref;
  Runner m_default_runner;
  Ref<RunQueue> m_default_queue_ref;
  RunQueue m_default_queue;
  bool m_has_init = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMng::
initialize(const AcceleratorRuntimeInitialisationInfo& runtime_info)
{
  if (m_has_init)
    ARCANE_FATAL("Method initialize() has already been called");

  arcaneInitializeRunner(m_default_runner,traceMng(),runtime_info);
  m_has_init = true;

  m_default_runner_ref = std::make_unique<Runner>(m_default_runner);
  m_default_queue_ref = makeQueueRef(m_default_runner);
  m_default_queue = *m_default_queue_ref.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT Ref<IAcceleratorMng>
arcaneCreateAcceleratorMngRef(ITraceMng* tm)
{
  return makeRef<IAcceleratorMng>(new AcceleratorMng(tm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
