// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivitySynchronizer.h                             (C) 2000-2015 */
/*                                                                           */
/* Interface de synchronisation de la connectivité des entités.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMCONNECTIVITYSYNCHRONIZER_H
#define ARCANE_IITEMCONNECTIVITYSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IItemConnectivitySynchronizer
{
 public:

  /** Destructeur de la classe */
  virtual ~IItemConnectivitySynchronizer() {}

 public:

  virtual IItemConnectivity* getConnectivity() = 0;

  virtual void synchronize() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ICONNECTIVITYSYNCHRONIZER_H */
