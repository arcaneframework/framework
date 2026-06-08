// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ViewBuildInfo.h                                             (C) 2000-2025 */
/*                                                                           */
/* Information to build a view for accelerator data.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_VIEWBUILDINFO_H
#define ARCCORE_COMMON_ACCELERATOR_VIEWBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information to build a view for accelerator data.
 *
 * Instances of this class are temporary and should not be
 * kept beyond the lifetime of the RunCommand or RunQueue used
 * for their creation.
 */
class ARCCORE_COMMON_EXPORT ViewBuildInfo
{
  friend class NumArrayViewBase;
  friend class VariableViewBase;

 public:

  // NOTE: the following constructors must be implicit

  //! Create instance associated with the queue.
  ViewBuildInfo(const RunQueue& queue);
  //! Create instance associated with the queue.
  ViewBuildInfo(const RunQueue* queue);
  //! Create instance associated with the command.
  ViewBuildInfo(RunCommand& command);

 private:

  Impl::RunQueueImpl* _internalQueue() const { return m_queue_impl; }

 private:

  Impl::RunQueueImpl* m_queue_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
