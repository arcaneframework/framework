// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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
inline TimeMetricActionHandle _build(ITimeMetricCollector* c,TimeMetricPhase p)
{
  return TimeMetricActionHandle(c,TimeMetricActionHandleBuildInfo(String(),(int)p));
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
