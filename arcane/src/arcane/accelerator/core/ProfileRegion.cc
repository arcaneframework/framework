// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfileRegion.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Région pour le profiling.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/ProfileRegion.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfileRegion::
ProfileRegion(const RunQueue& queue, const String& name)
{
  if (queue.isNull())
    return;
  m_runtime = queue._internalRuntime();
  m_runtime->pushProfilerRange(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfileRegion::
~ProfileRegion()
{
  if (m_runtime)
    m_runtime->popProfilerRange();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
