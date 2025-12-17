// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ForLoopTraceInfo.h                                          (C) 2000-2025 */
/*                                                                           */
/* Informations de trace pour une boucle for.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FORLOOPTRACEINFO_H
#define ARCCORE_BASE_FORLOOPTRACEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de trace pour une boucle 'for'.
 */
class ARCCORE_BASE_EXPORT ForLoopTraceInfo
{
 public:

  ForLoopTraceInfo() = default;
  explicit ForLoopTraceInfo(const TraceInfo& trace_info)
  : m_trace_info(trace_info)
  , m_is_valid(true)
  {
  }
  ForLoopTraceInfo(const TraceInfo& trace_info, const String& loop_name)
  : m_trace_info(trace_info)
  , m_loop_name(loop_name)
  , m_is_valid(true)
  {
  }

 public:

  const TraceInfo& traceInfo() const { return m_trace_info; }
  const String& loopName() const { return m_loop_name; }
  bool isValid() const { return m_is_valid; }

 private:

  TraceInfo m_trace_info;
  String m_loop_name;
  bool m_is_valid = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

