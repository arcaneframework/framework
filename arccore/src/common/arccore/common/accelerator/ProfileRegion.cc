// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfileRegion.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Région pour le profiling.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/ProfileRegion.h"

#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/accelerator/internal/IRunnerRuntime.h"

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
  m_runtime->pushProfilerRange(name, -1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfileRegion::
ProfileRegion(const RunQueue& queue, const String& name, Int32 color_rgb)
{
  if (queue.isNull())
    return;
  m_runtime = queue._internalRuntime();
  m_runtime->pushProfilerRange(name, color_rgb);
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
