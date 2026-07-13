// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Limits.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Files encapsulating <limits> and associated types.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LIMITS_H
#define ARCANE_UTILS_LIMITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#ifndef ARCCORE_COMPILING_FRAMEWORK
#include "arcane/utils/StdHeader.h"
#endif

// Since <limits> defines min, max, abs, ... and some software
// makes them macros, we remove them
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef abs
#undef abs
#endif
#include <limits>

#include "arccore/base/FloatInfo.h"
#include <float.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
