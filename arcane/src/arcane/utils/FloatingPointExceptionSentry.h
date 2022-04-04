﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FloatingPointExceptionSentry.h                              (C) 2000-2017 */
/*                                                                           */
/* Activation/désactivation temporaire des exceptions flottantes             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FLOATINGPOINTEXCEPTIONSENTRY_H
#define ARCANE_UTILS_FLOATINGPOINTEXCEPTIONSENTRY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe permettant d'activer/désactiver temporairement les exceptions
 * flottantes du processeur.
 *
 * Cette classe permet de modifier temporairement l'état des exceptions
 * flottantes. L'ancien état est réactivé lors de l'appel au destructeur.
 */
class ARCANE_UTILS_EXPORT FloatingPointExceptionSentry
{
 public:

  explicit FloatingPointExceptionSentry(bool want_active);
  ~FloatingPointExceptionSentry();

 private:

  bool m_want_active;
  bool m_is_active;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

