// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProcessorAffinityService.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface of a CPU core affinity management service.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
#define ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a CPU core affinity management service.
 */
class IProcessorAffinityService
{
 public:

  virtual ~IProcessorAffinityService() {} //<! Frees resources

 public:

  virtual void build() = 0;

 public:

  //! Displays complete topology information via info()
  virtual void printInfos() = 0;

  /*!
   * \brief Returns the cpuset for the current thread.
   *
   * The returned string is in a format compatible with that of
   * taskset. For example, we can have values such as
   * 'ff', '1, or 'ffff1234,ff'.
   */
  virtual String cpuSetString() = 0;

  //! Constrains the current thread to stay on the core with index \a cpu
  virtual void bindThread(Int32 cpu) = 0;

  //! Number of CPU cores (-1 if unknown)
  virtual Int32 numberOfCore() = 0;

  //! Number of sockets (-1 if unknown)
  virtual Int32 numberOfSocket() = 0;

  //! Number of logical cores (-1 if unknown)
  virtual Int32 numberOfProcessingUnit() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
