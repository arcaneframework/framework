﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIEN_TRILINOSIMPL_TRILINOSPRECOMP_H
#define ALIEN_TRILINOSIMPL_TRILINOSPRECOMP_H
/* Author : mesriy at Tue Jul 24 15:49:46 2012
 * Generated by createNew
 */

#include <alien/utils/Precomp.h>
#include <alien/AlienTrilinosPrecomp.h>
#include "Trilinos_version.h"


#if (TRILINOS_MAJOR_VERSION < 15)
#if __cplusplus >= 202002L
// C++20 (and later) code
//#define HAVE_MUELU  MUELU DESACTIVATED
#else
#define HAVE_MUELU
#endif
#else
#define HAVE_MUELU
#endif


#define BEGIN_TRILINOSINTERNAL_NAMESPACE                                                 \
  namespace Alien {                                                                      \
  namespace TrilinosInternal {
#define END_TRILINOSINTERNAL_NAMESPACE                                                   \
  }                                                                                      \
  }

#endif /* ALIEN_TRILINOSIMPL_TRILINOSPRECOMP_H */
