// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelExchangerOptions.h                                  (C) 2000-2025 */
/*                                                                           */
/* Options to modify the behavior of 'IParallelExchanger'.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLELEXCHANGEROPTIONS_H
#define ARCANE_CORE_PARALLELEXCHANGEROPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Options for IParallelMng::processExchange().
 */
class ARCANE_CORE_EXPORT ParallelExchangerOptions
{
 public:
  /*!
   * \brief Exchange mode.
   */
  enum eExchangeMode
  {
    //! Uses point-to-point exchanges (send/recv)
    EM_Independant,
    //! Uses collective operations (allToAll)
    EM_Collective,
    //! Automatically chooses between point-to-point or collective.
    EM_Auto
  };

 public:

  //! Sets the exchange mode.
  void setExchangeMode(eExchangeMode mode) { m_exchange_mode = mode; }
  //! Specified exchange mode
  eExchangeMode exchangeMode() const { return m_exchange_mode; };

  //! Sets the maximum number of pending messages.
  void setMaxPendingMessage(Int32 v) { m_max_pending_message = v; }
  //! Maximum number of pending messages
  Int32 maxPendingMessage() const { return m_max_pending_message; };

  //! Sets the verbosity level
  void setVerbosityLevel(Int32 v) { m_verbosity_level = v; }
  //! Verbosity level
  Int32 verbosityLevel() const { return m_verbosity_level; };

 private:

  //! Exchange mode.
  eExchangeMode m_exchange_mode = EM_Independant;

  //! Maximum number of pending messages
  Int32 m_max_pending_message = 0;

  //! Verbosity level
  Int32 m_verbosity_level = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
