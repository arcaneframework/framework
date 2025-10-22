// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ForLoopRunInfo.h                                            (C) 2000-2025 */
/*                                                                           */
/* Informations d'exécution d'une boucle.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FORLOOPRUNINFO_H
#define ARCCORE_BASE_FORLOOPRUNINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ForLoopTraceInfo.h"

#include "arccore/concurrency/ParallelLoopOptions.h"

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations d'exécution d'une boucle.
 *
 * Cette classe permet de gérer les informations d'exécutions communes à toutes
 * les boucles.
 */
class ARCCORE_CONCURRENCY_EXPORT ForLoopRunInfo
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
   * \brief Positionne le pointeur conservant les statistiques d'exécution.
   *
   * Ce pointeur \a v doit rester valide durant toute l'exécution de la boucle.
   */
  void setExecStat(ForLoopOneExecStat* v) { m_exec_stat = v; }

  //! Pointeur contenant les statistiques d'exécution.
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
