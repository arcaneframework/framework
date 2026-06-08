// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceGlobal.h                                               (C) 2000-2025 */
/*                                                                           */
/* Global definitions for the 'Trace' component of 'Arccore'.                */
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

} // namespace Arcane

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

using Arcane::ITimeMetricCollector;
using Arcane::ITraceMessageListener;
using Arcane::ITraceMng;
using Arcane::ITraceStream;
using Arcane::StandaloneTraceMessage;
using Arcane::TimeMetricAction;
using Arcane::TimeMetricActionBuildInfo;
using Arcane::TimeMetricId;
using Arcane::TimeMetricSentry;
using Arcane::TraceAccessor;
using Arcane::TraceClassConfig;
using Arcane::TraceMessage;
using Arcane::TraceMessageClass;
using Arcane::TraceMessageDbg;
using Arcane::TraceMessageListenerArgs;
namespace Trace = ::Arcane::Trace;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
