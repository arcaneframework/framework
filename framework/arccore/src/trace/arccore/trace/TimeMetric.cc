// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeMetric.h                                                (C) 2000-2020 */
/*                                                                           */
/* Classes gérant les métriques temporelles.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TimeMetric.h"
#include "arccore/trace/ITimeMetricCollector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
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
