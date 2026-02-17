// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngInternal.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Implémentation de la partie interne à Arcane de IParallelMng.             */
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/ParallelMngInternal.h"

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"

#include "arcane/core/ParallelMngDispatcher.h"

#include "arcane/utils/Convert.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngInternal::
ParallelMngInternal(ParallelMngDispatcher* pm)
: m_parallel_mng(pm)
, m_runner(Accelerator::eExecutionPolicy::Sequential)
, m_queue(makeQueue(m_runner))
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DISABLE_ACCELERATOR_AWARE_MESSAGE_PASSING", true))
    m_is_accelerator_aware_disabled = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner ParallelMngInternal::
runner() const
{
  return m_runner;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueue ParallelMngInternal::
queue() const
{
  return m_queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParallelMngInternal::
isAcceleratorAware() const
{
  if (m_is_accelerator_aware_disabled)
    return false;
  if (m_queue.isNull())
    return false;
  if (!m_queue.isAcceleratorPolicy())
    return false;
  return m_parallel_mng->_isAcceleratorAware();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngInternal::
setDefaultRunner(const Runner& runner)
{
  if (!m_runner.isInitialized())
    ARCANE_FATAL("Can not set an unitialized Runner");

  // Attention à bien supprimer la référence sur la RunQueue
  // avant de détruire le Runner car s'il n'y a pas d'autres
  // références sur \a m_runner il sera détruit avec \a m_queue
  // et ce dernier aura un \a m_runner détruit.
  m_queue = RunQueue{};
  m_runner = runner;
  Accelerator::RunQueueBuildInfo build_info(-5);
  m_queue = makeQueue(m_runner, build_info);
  m_queue.setAsync(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> ParallelMngInternal::
createSubParallelMngRef(Int32 color, Int32 key)
{
  return m_parallel_mng->_createSubParallelMngRef(color, key);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<MessagePassing::IMachineMemoryWindowBaseInternal> ParallelMngInternal::
createMachineMemoryWindowBase([[maybe_unused]] Int64 sizeof_segment,
                              [[maybe_unused]] Int32 sizeof_type)
{
  ARCANE_THROW(NotImplementedException, "MachineWindow is not available in your ParallelMng");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<MessagePassing::IDynamicMachineMemoryWindowBaseInternal> ParallelMngInternal::
createDynamicMachineMemoryWindowBase([[maybe_unused]] Int64 sizeof_segment,
                                     [[maybe_unused]] Int32 sizeof_type)
{
  ARCANE_THROW(NotImplementedException, "MachineWindow is not available in your ParallelMng");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* ParallelMngInternal::dynamicMachineMemoryWindowMemoryAllocator()
{
  ARCANE_THROW(NotImplementedException, "MachineWindow is not available in your ParallelMng");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
