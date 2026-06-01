// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdHeader.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Standard header files.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_STDHEADER_H
#define ARCANE_UTILS_STDHEADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <locale>

// Temporary (April 2024) for compatibility with existing code.
// Later, these using statements must be removed
#if !defined(ARCANE_NO_USE_STD_MATH_FUNCTIONS)
using std::abs;
using std::cos;
using std::isinf;
using std::isnan;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
