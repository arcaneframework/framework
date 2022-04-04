// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdHeader.h                                                 (C) 2000-2005 */
/*                                                                           */
/* Fichiers d'entêtes standards.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_STDHEADER_H
#define ARCANE_UTILS_STDHEADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#ifdef ARCANE_USE_STD_HEADER
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <locale>

#ifndef ARCANE_OS_TRU64
// Bug sur cette machine avec <ctype> (erreur de compilation dans la STL)
#include <ctype>
using std::isalpha;
using std::isdigit;
using std::tolower;
#endif

using std::memcpy;
using std::memcmp;
using std::memset;
using std::strchr;
using std::strcmp;
using std::strtod;
using std::strtoul;
using std::strtol;
using std::strcpy;
using std::strlen;
using std::strncpy;
using std::strcat;
using std::abort;
using std::exit;
using std::rand;

// A partir de <cstdlib>
using std::malloc;
using std::realloc;
using std::free;
using std::memmove;

// A partir de <cmath>
using std::fabs;
using std::sqrt;
using std::acos;
using std::asin;
using std::atan;
using std::ceil;
using std::cos;
using std::cosh;
using std::exp;
using std::floor;
using std::log;
using std::log10;
using std::sin;
using std::sinh;
using std::tan;
using std::tanh;
using std::pow;

using std::fabsl;
using std::sqrtl;
using std::acosl;
using std::asinl;
using std::atanl;
using std::ceill;
using std::cosl;
using std::coshl;
using std::expl;
using std::floorl;
using std::logl;
using std::log10l;
using std::sinl;
using std::sinhl;
using std::tanl;
using std::tanhl;
using std::powl;

using std::fabsf;
using std::sqrtf;
using std::acosf;
using std::asinf;
using std::atanf;
using std::ceilf;
using std::cosf;
using std::coshf;
using std::expf;
using std::floorf;
using std::logf;
using std::log10f;
using std::sinf;
using std::sinhf;
using std::tanf;
using std::tanhf;
using std::powf;
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#endif

#ifdef ARCANE_OS_TRU64
#include <ctype.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif




















