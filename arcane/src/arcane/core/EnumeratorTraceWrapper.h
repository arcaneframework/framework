// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnumeratorTraceWrapper.h                                    (C) 2000-2024 */
/*                                                                           */
/* Enumerator over groups of mesh entities.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ENUMERATORTRACEWRAPPER_H
#define ARCANE_ENUMERATORTRACEWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/TraceInfo.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_TRACE_ENUMERATOR
// Uncomment if you want to always enable enumerator traces
//#define ARCANE_TRACE_ENUMERATOR
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for an enumerator's traces.
 */
class EnumeratorTraceInfo
{
 public:

  //! Hardware counter values.
  Int64ArrayView counters() { return Int64ArrayView(8, m_counters.data()); }

  //! Start time in nanoseconds
  Int64 beginTime() const { return m_begin_time; }

  //! Sets the start time
  void setBeginTime(Int64 v) { m_begin_time = v; }

  //! Sets the trace information
  void setTraceInfo(const TraceInfo* ti)
  {
    if (ti) {
      m_trace_info = *ti;
      m_has_trace_info = true;
    }
    else
      m_has_trace_info = false;
  }

  //! Trace information (or nullptr) if none
  const TraceInfo* traceInfo() const
  {
    return (m_has_trace_info) ? &m_trace_info : nullptr;
  }

 private:

  std::array<Int64, 8> m_counters = {};
  Int64 m_begin_time = 0;
  TraceInfo m_trace_info;
  bool m_has_trace_info = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Wrapper around an enumerator for traces.
 *
 * \a TrueEnumerator is the type of the actual enumerator and
 * \a TraceInterface is that of the management interface. The type \a TraceInterface
 * must have the following properties:
 * - a singleton() method returning an instance.
 * - an enterEnumerator() method and an exitEnumerator() method for
 * each supported enumerator type.
 */
template <typename TrueEnumerator, typename TracerInterface>
class EnumeratorTraceWrapper
: public TrueEnumerator
{
 public:

  ARCCORE_HOST_DEVICE EnumeratorTraceWrapper(TrueEnumerator&& tenum)
  : TrueEnumerator(tenum)
  {
#ifndef ARCCORE_DEVICE_CODE
    m_tracer = TracerInterface::singleton();
    if (m_tracer)
      m_tracer->enterEnumerator(*this, m_infos);
#endif
  }
  ARCCORE_HOST_DEVICE EnumeratorTraceWrapper(TrueEnumerator&& tenum, [[maybe_unused]] const TraceInfo& ti)
  : TrueEnumerator(tenum)
  {
#ifndef ARCCORE_DEVICE_CODE
    m_tracer = TracerInterface::singleton();
    if (m_tracer) {
      m_infos.setTraceInfo(&ti);
      m_tracer->enterEnumerator(*this, m_infos);
    }
#endif
  }
  ARCCORE_HOST_DEVICE ~EnumeratorTraceWrapper() ARCANE_NOEXCEPT_FALSE
  {
#ifndef ARCCORE_DEVICE_CODE
    if (m_tracer)
      m_tracer->exitEnumerator(*this, m_infos);
#endif
  }

 private:

  TracerInterface* m_tracer = nullptr;
  EnumeratorTraceInfo m_infos;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_TRACE_ENUMERATOR)
#define A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) \
  ::Arcane::EnumeratorTraceWrapper<_EnumeratorClassName, ::Arcane::IItemEnumeratorTracer>
#define A_TRACE_ENUMERATOR_WHERE , A_FUNCINFO
#else
#define A_TRACE_ITEM_ENUMERATOR(_EnumeratorClassName) \
  _EnumeratorClassName
#define A_TRACE_ENUMERATOR_WHERE
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
