// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Initializer.h                                               (C) 2000-2026 */
/*                                                                           */
/* Class to initialize the accelerator runtime.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_INTERNAL_INITIALIZER_H
#define ARCCORE_ACCELERATOR_INTERNAL_INITIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#include "arccore/trace/ITraceMng.h"

#include "arccore/common/ArccoreApplicationBuildInfo.h"
#include "arccore/concurrency/internal/ConcurrencyApplication.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Internal class to initialize the accelerator runtime.
 */
class ARCCORE_ACCELERATOR_EXPORT Initializer
{
 public:

  Initializer(bool use_accelerator, Int32 max_allowed_thread);
  ~Initializer() noexcept(false);

 public:

  Initializer(const Initializer&) = delete;
  Initializer(Initializer&&) = delete;
  Initializer& operator=(const Initializer&) = delete;
  Initializer& operator=(Initializer&&) = delete;

 public:

  eExecutionPolicy executionPolicy() const { return m_policy; }
  ITraceMng* traceMng() const { return m_trace_mng.get(); }

 private:

  eExecutionPolicy m_policy = eExecutionPolicy::Sequential;
  ReferenceCounter<ITraceMng> m_trace_mng;
  ArccoreApplicationBuildInfo m_application_build_info;
  ConcurrencyApplication m_concurrency_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
