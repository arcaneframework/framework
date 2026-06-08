// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Managing views on 'NumArray' for accelerators.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/NumArrayViews.h"

#include "arccore/base/MemoryView.h"

#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/common/accelerator/Memory.h"
#include "arccore/common/accelerator/internal/RunQueueImpl.h"
#include "arccore/common/accelerator/internal/IRunQueueStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file NumArrayViews.h
 *
 * This file contains the type declarations for managing
 * views for 'NumArray' class accelerators.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NumArrayViewBase::
NumArrayViewBase(const ViewBuildInfo& vbi, Span<const std::byte> bytes)
{
  Impl::RunQueueImpl* q = vbi._internalQueue();
  if (q->isAutoPrefetchCommand()) {
    ConstMemoryView mem_view(bytes);
    q->prefetchMemory(MemoryPrefetchArgs(mem_view).addAsync());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
