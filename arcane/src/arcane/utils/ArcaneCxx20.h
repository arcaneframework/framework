// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCxx20.h                                               (C) 2000-2023 */
/*                                                                           */
/* Vérification des fonctionnalités minimale requises du C++20.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARCANECXX20_H
#define ARCANE_UTILS_ARCANECXX20_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_HAS_CXX20
#error "You need to compile Arcane with C++20 support to use this header file."
"Add -DARCCORE_CXX_STANDARD=20 to build configuration"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
