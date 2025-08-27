// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelExchangerOptions.h                                  (C) 2000-2022 */
/*                                                                           */
/* Options pour modifier le comportement de 'IParallelExchanger'.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLELEXCHANGEROPTIONS_H
#define ARCANE_PARALLELEXCHANGEROPTIONS_H
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
 * \brief Options pour IParallelMng::processExchange().
 */
class ARCANE_CORE_EXPORT ParallelExchangerOptions
{
 public:
  /*!
   * \brief Mode d'échange.
   */
  enum eExchangeMode
  {
    //! Utilise les échanges point à point (send/recv)
    EM_Independant,
    //! Utilise les opération collectives (allToAll)
    EM_Collective,
    //! Choisi automatiquement entre point à point ou collective.
    EM_Auto
  };

 public:

  //! Positionne le mode d'échange.
  void setExchangeMode(eExchangeMode mode) { m_exchange_mode = mode; }
  //! Mode d'échange spécifié
  eExchangeMode exchangeMode() const { return m_exchange_mode; };

  //! Positionne le nombre maximal de messages en vol.
  void setMaxPendingMessage(Int32 v) { m_max_pending_message = v; }
  //! Nombre maximal de messages en vol
  Int32 maxPendingMessage() const { return m_max_pending_message; };

  //! Positionne le niveau de verbosité
  void setVerbosityLevel(Int32 v) { m_verbosity_level = v; }
  //! Niveau de verbosité
  Int32 verbosityLevel() const { return m_verbosity_level; };

 private:

  //! Mode d'échange.
  eExchangeMode m_exchange_mode = EM_Independant;

  //! Nombre maximal de messages en vol
  Int32 m_max_pending_message = 0;

  //! Niveau de verbosité
  Int32 m_verbosity_level = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
