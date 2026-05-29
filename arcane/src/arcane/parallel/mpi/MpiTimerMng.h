// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTimerMng.h                                               (C) 2000-2006 */
/*                                                                           */
/* Timer manager using MPI_Wtime.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPITIMERMNG_H
#define ARCANE_PARALLEL_MPI_MPITIMERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/TimerMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Timer manager using the MPI library.
 *
 * This timer functions the same way as the base class TimerMng
 * except for how it calculates real time, which uses the
 * MPI_Wtime() function.
 *
 \since 0.8.0
 \author Gilles Grospellier
 \date 05/09/2001
 */
class MpiTimerMng
: public TimerMng
{
 public:

  //! Constructs a timer linked to the subdomain \a mng
  MpiTimerMng(ITraceMng* trace);

 public:

  virtual ~MpiTimerMng();

 protected:

  virtual Real _getRealTime();
  virtual void _setRealTime();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
