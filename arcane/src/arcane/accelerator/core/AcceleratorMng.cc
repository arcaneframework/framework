// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorMng.cc                                           (C) 2000-2021 */
/*                                                                           */
/* Implémentation de 'IAcceleratorMng'                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Ref.h"

#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/RunQueue.h"

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

  AcceleratorMng(ITraceMng* tm)
  : TraceAccessor(tm)
  {
    m_default_runner = makeRef(new Runner());
  }

 public:

  void initialize() override;
  Runner* defaultRunner() override { return m_default_runner.get(); }
  RunQueue* defaultQueue() override { return m_default_queue.get(); }

 private:

  Ref<Runner> m_default_runner;
  Ref<RunQueue> m_default_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorMng::
initialize()
{
  m_default_queue = makeQueueRef(*(m_default_runner.get()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT Ref<IAcceleratorMng>
createAcceleratorMngRef(ITraceMng* tm)
{
  return makeRef<IAcceleratorMng>(new AcceleratorMng(tm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
