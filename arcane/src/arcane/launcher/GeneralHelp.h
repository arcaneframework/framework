// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeneralHelp.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classe gérant le message d'aide générique.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_GENERALHELP_H
#define ARCANE_LAUNCHER_GENERALHELP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

namespace Arcane
{
class ARCANE_LAUNCHER_EXPORT GeneralHelp
{
public:
  static void printHelp();
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
