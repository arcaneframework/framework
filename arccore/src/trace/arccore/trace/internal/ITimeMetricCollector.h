// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeMetricCollector.h                                      (C) 2000-2025 */
/*                                                                           */
/* Interface managing statistics on execution times.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_ITIMEMETRICCOLLECTOR_H
#define ARCCORE_TRACE_ITIMEMETRICCOLLECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"
#include "arccore/base/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * API under definition. Do not use outside of Arccore/Arcane.
 */

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface managing statistics on execution.
 */
class ARCCORE_TRACE_EXPORT ITimeMetricCollector
{
 public:
 public:

  // Releases resources.
  virtual ~ITimeMetricCollector() = default;

 public:

  virtual TimeMetricAction getAction(const TimeMetricActionBuildInfo& x) = 0;
  virtual TimeMetricId beginAction(const TimeMetricAction& handle) = 0;
  virtual void endAction(const TimeMetricId& metric_id) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
