// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimerMng.h                                                  (C) 2000-2019 */
/*                                                                           */
/* Implémentation d'un gestionnaire de timer.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_TIMERMNG_H
#define ARCANE_IMPL_TIMERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ITimerMng.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestionnaire de timer.
 *
 * \warning Cette classe est interne à Arcane.
 */
class ARCANE_IMPL_EXPORT TimerMng
: public TraceAccessor
, public ITimerMng
{
 public:

  //! Construit un timer lié au gestionnaire \a mng
  explicit TimerMng(ITraceMng* msg);

 public:

  ~TimerMng() override;

 public:

  void beginTimer(Timer* timer) override;
  Real endTimer(Timer* timer) override;
  Real getTime(Timer* timer) override;
  bool hasTimer(Timer* timer) override;

 protected:

  //! Retourne le temps réel
  virtual Real _getRealTime();

  //! Positionne un timer réel
  virtual void _setRealTime();
  
 private:

  std::atomic<Int64> m_nb_timer;

 private:

  void _errorInTimer(const String& msg,int retcode);

  void _setVirtualTime();
  Real _getVirtualTime();

  void _setTime(Timer*);
  Real _getTime(Timer*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

