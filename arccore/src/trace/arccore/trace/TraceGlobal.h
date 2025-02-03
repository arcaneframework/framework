// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceGlobal.h                                               (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Trace' de 'Arccore'.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TRACEGLOBAL_H
#define ARCCORE_TRACE_TRACEGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_trace)
#define ARCCORE_TRACE_EXPORT ARCCORE_EXPORT
#define ARCCORE_TRACE_EXTERN_TPL
#else
#define ARCCORE_TRACE_EXPORT ARCCORE_IMPORT
#define ARCCORE_TRACE_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
class ITraceStream;
class ITraceMng;
class TraceMessageClass; 
class TraceClassConfig;
class TraceMessage;
class TraceMessageListenerArgs;
class ITraceMessageListener;
class TraceAccessor;
class StandaloneTraceMessage;
#ifdef ARCCORE_DEBUG
typedef TraceMessage TraceMessageDbg;
#else
class TraceMessageDbg;
#endif

class ITimeMetricCollector;
class TimeMetricSentry;
class TimeMetricId;
class TimeMetricAction;
class TimeMetricActionBuildInfo;

namespace Trace
{
}

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(ITraceStream)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(ITraceMng)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arccore::ITraceStream;
using Arccore::ITraceMng;
using Arccore::TraceAccessor;
using Arccore::TraceMessageClass; 
using Arccore::TraceClassConfig;
using Arccore::TraceMessage;
using Arccore::TraceMessageDbg;
using Arccore::TraceMessageListenerArgs;
using Arccore::ITraceMessageListener;
using Arccore::ITimeMetricCollector;
using Arccore::TimeMetricSentry;
using Arccore::TimeMetricId;
using Arccore::TimeMetricAction;
using Arccore::TimeMetricActionBuildInfo;
namespace Trace = ::Arccore::Trace;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

