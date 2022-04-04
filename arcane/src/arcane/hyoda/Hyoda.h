// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * hyoda.h                                                     (C) 2000-2012 *
 *****************************************************************************/
#ifndef ARCANE_HYODA_H
#define ARCANE_HYODA_H

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IOnlineDebuggerService.h"

/*
 * Macros utilisées pour Hyoda
 */
#ifdef __GNUG__
# define ARCANE_HYODA_SOFTBREAK(subDomain) {                            \
    IOnlineDebuggerService *hyoda=platform::getOnlineDebuggerService(); \
    if (hyoda) hyoda->softbreak(subDomain,                              \
                                __FILE__,                               \
                                __PRETTY_FUNCTION__,                    \
                                __LINE__);                              \
  }
#else
#  define ARCANE_HYODA_SOFTBREAK(subDomain) {                           \
    IOnlineDebuggerService *hyoda=platform::getOnlineDebuggerService(); \
    if (hyoda) hyoda->softbreak(subDomain,\
                                __FILE__,\
                                "(NoInfo)",\
                                __LINE__);\
  }
#endif

# define ARCANE_HYODA_SOFTBREAK_IF(bool, subDomain) {                 \
    if (bool==true){ARCANE_HYODA_SOFTBREAK(subDomain);}               \
  }


#endif //  ARCANE_HYODA_H
