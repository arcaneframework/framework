// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimerMng.h                                                  (C) 2000-2022 */
/*                                                                           */
/* Implementation of a timer manager.                                        */
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
 * \brief Timer manager.
 *
 * \warning This class is internal to Arcane.
 */
class ARCANE_IMPL_EXPORT TimerMng
: public TraceAccessor
, public ITimerMng
{
 public:

  //! Constructs a timer linked to the manager \a mng
  explicit TimerMng(ITraceMng* msg);

 public:

  void beginTimer(Timer* timer) override;
  Real endTimer(Timer* timer) override;
  Real getTime(Timer* timer) override;
  bool hasTimer(Timer* timer) override;

 protected:

  //! Returns the real time
  virtual Real _getRealTime();

  //! Sets the real time
  virtual void _setRealTime() {}
  
 private:

  void _errorInTimer(const String& msg,int retcode);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
