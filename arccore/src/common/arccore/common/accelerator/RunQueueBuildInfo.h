// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueBuildInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Information for creating a RunQueue.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNQUEUEBUILDINFO_H
#define ARCCORE_COMMON_ACCELERATOR_RUNQUEUEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information to create a RunQueue.
 */
class ARCCORE_COMMON_EXPORT RunQueueBuildInfo
{
 public:

  RunQueueBuildInfo() = default;
  explicit RunQueueBuildInfo(int priority)
  : m_priority(priority)
  {}

 public:

  /*!
  * \brief Sets the priority.
  *
  * By default, the priority is 0, which indicates that a 'RunQueue'
  * is created with the default priority. Strictly positive values indicate
  * a lower priority and strictly negative values indicate a higher priority.
  */
  void setPriority(int priority) { m_priority = priority; }
  int priority() const { return m_priority; }

  //! Indicates if the instance only has default values.
  bool isDefault() const { return m_priority == 0; }

 private:

  int m_priority = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
