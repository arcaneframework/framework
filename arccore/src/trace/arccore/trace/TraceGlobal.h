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

namespace Arcane
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

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ITraceStream)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ITraceMng)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using Arcane::ITraceStream;
using Arcane::ITraceMng;
using Arcane::TraceAccessor;
using Arcane::TraceMessageClass; 
using Arcane::TraceClassConfig;
using Arcane::TraceMessage;
using Arcane::TraceMessageDbg;
using Arcane::TraceMessageListenerArgs;
using Arcane::ITraceMessageListener;
using Arcane::ITimeMetricCollector;
using Arcane::TimeMetricSentry;
using Arcane::TimeMetricId;
using Arcane::TimeMetricAction;
using Arcane::TimeMetricActionBuildInfo;
using Arcane::StandaloneTraceMessage;
namespace Trace = ::Arcane::Trace;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

