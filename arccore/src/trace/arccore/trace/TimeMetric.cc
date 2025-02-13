// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeMetric.h                                                (C) 2000-2025 */
/*                                                                           */
/* Classes gérant les métriques temporelles.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TimeMetric.h"
#include "arccore/trace/ITimeMetricCollector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
inline TimeMetricAction _build(ITimeMetricCollector* c,TimeMetricPhase p)
{
  if (c)
    return TimeMetricAction(c,TimeMetricActionBuildInfo(String(),(int)p));
  return TimeMetricAction();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardPhaseTimeMetrics::
initialize(ITimeMetricCollector* collector)
{
  if (!collector)
    return;
  m_message_passing_phase = _build(collector,TimeMetricPhase::MessagePassing);
  m_input_output_phase = _build(collector,TimeMetricPhase::InputOutput);
  m_computation_phase = _build(collector,TimeMetricPhase::Computation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricAction
timeMetricPhaseMessagePassing(ITimeMetricCollector* c)
{
  return _build(c,TimeMetricPhase::MessagePassing);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricAction
timeMetricPhaseInputOutput(ITimeMetricCollector* c)
{
  return _build(c,TimeMetricPhase::InputOutput);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricAction
timeMetricPhaseComputation(ITimeMetricCollector* c)
{
  return _build(c,TimeMetricPhase::Computation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
