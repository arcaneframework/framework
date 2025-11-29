// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueBuildInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Informations pour créer une RunQueue.                                     */
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
 * \brief Informations pour créer une RunQueue.
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
  * \brief Positionne la priorité.
  *
  * Par défaut la priorité vaut 0 et cela indique qu'on créé une 'RunQueue'
  * avec la priorité par défaut. Les valeurs strictement positives indiquent
  * une priorité plus faible et les valeurs strictement négatives une priorité
  * plus élevée.
  */
  void setPriority(int priority) { m_priority = priority; }
  int priority() const { return m_priority; }

  //! Indique si l'instance a uniquement les valeurs par défaut.
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
