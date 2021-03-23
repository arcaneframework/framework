// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTimerMng.h                                               (C) 2000-2006 */
/*                                                                           */
/* Gestionnaire de timer utilisant MPI_Wtime.                                */
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
 * \brief Gestionnaire de timer utisant la bibliothèque MPI.
 *
 * Ce timer fonctionne de la même manière que celui de la classe de base TimerMng
 * sauf pour la manière de calculer le temps réel, qui utilise la
 * fonction MPI_Wtime().
 *
 \since 0.8.0
 \author Gilles Grospellier
 \date 05/09/2001
 */
class MpiTimerMng
: public TimerMng
{
 public:

  //! Construit un timer lié au sous-domaine \a mng
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

