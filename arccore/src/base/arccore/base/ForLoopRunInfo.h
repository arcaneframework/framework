// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ForLoopRunInfo.h                                            (C) 2000-2025 */
/*                                                                           */
/* Loop execution information.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FORLOOPRUNINFO_H
#define ARCCORE_BASE_FORLOOPRUNINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ForLoopTraceInfo.h"
#include "arccore/base/ParallelLoopOptions.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Loop execution information.
 *
 * This class allows managing execution information common to all
 * loops.
 */
class ARCCORE_BASE_EXPORT ForLoopRunInfo
{
 public:

  using ThatClass = ForLoopRunInfo;

 public:

  ForLoopRunInfo() = default;
  explicit ForLoopRunInfo(const ParallelLoopOptions& options)
  : m_options(options)
  {}
  ForLoopRunInfo(const ParallelLoopOptions& options, const ForLoopTraceInfo& trace_info)
  : m_options(options)
  , m_trace_info(trace_info)
  {}
  explicit ForLoopRunInfo(const ForLoopTraceInfo& trace_info)
  : m_trace_info(trace_info)
  {}

 public:

  std::optional<ParallelLoopOptions> options() const { return m_options; }
  ThatClass& addOptions(const ParallelLoopOptions& v)
  {
    m_options = v;
    return (*this);
  }
  const ForLoopTraceInfo& traceInfo() const { return m_trace_info; }
  ThatClass& addTraceInfo(const ForLoopTraceInfo& v)
  {
    m_trace_info = v;
    return (*this);
  }

  /*!
   * \brief Sets the pointer holding the execution statistics.
   *
   * This pointer \a v must remain valid throughout the loop execution.
   */
  void setExecStat(ForLoopOneExecStat* v) { m_exec_stat = v; }

  //! Pointer containing execution statistics.
  ForLoopOneExecStat* execStat() const { return m_exec_stat; }

 protected:

  std::optional<ParallelLoopOptions> m_options;
  ForLoopTraceInfo m_trace_info;
  ForLoopOneExecStat* m_exec_stat = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
