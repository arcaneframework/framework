// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LauncherGlobal.h                                            (C) 2000-2019 */
/*                                                                           */
/* Déclarations générales de la composante 'Launcher' de Arcane.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_LAUNCHERGLOBAL_H
#define ARCANE_LAUNCHER_LAUNCHERGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_launcher
#define ARCANE_LAUNCHER_EXPORT ARCANE_EXPORT
#else
#define ARCANE_LAUNCHER_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ArcaneLauncher;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
