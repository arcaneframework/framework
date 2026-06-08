// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConcurrencyBase.h                                           (C) 2000-2025 */
/*                                                                           */
/* Base classes for multi-threading management.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CONCURRENCYBASE_H
#define ARCCORE_BASE_CONCURRENCYBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ParallelLoopOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Basic information for multi-threading management.
 */
class ARCCORE_BASE_EXPORT ConcurrencyBase
{
  // To call _setMaxAllowedThread.
  friend class TBBTaskImplementation;

 public:

  /*!
   * \brief Maximum number of allowed threads for multi-threading.
   *
   * This value is only meaningful once the management service
   * for multi-threading has been created.
   */
  static Int32 maxAllowedThread() { return m_max_allowed_thread; }

 public:

  //! Sets the default execution values for a parallel loop
  static void setDefaultParallelLoopOptions(const ParallelLoopOptions& v)
  {
    m_default_loop_options = v;
  }

  //! Default execution values for a parallel loop
  static const ParallelLoopOptions& defaultParallelLoopOptions()
  {
    return m_default_loop_options;
  }

 private:

  static Int32 m_max_allowed_thread;
  static ParallelLoopOptions m_default_loop_options;

 private:

  static void _setMaxAllowedThread(Int32 v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
