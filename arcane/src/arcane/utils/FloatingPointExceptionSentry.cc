// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FloatingPointExceptionSentry.cc                             (C) 2000-2017 */
/*                                                                           */
/* Activation/désactivation temporaire des exceptions flottantes             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FloatingPointExceptionSentry.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FloatingPointExceptionSentry::
FloatingPointExceptionSentry(bool want_active)
: m_want_active(want_active)
, m_is_active(platform::isFloatingExceptionEnabled())
{
  if (m_want_active!=m_is_active)
    platform::enableFloatingException(m_want_active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FloatingPointExceptionSentry::
~FloatingPointExceptionSentry()
{
  if (m_want_active!=m_is_active)
    platform::enableFloatingException(m_is_active);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
