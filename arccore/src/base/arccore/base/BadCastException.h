// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadCastException.h                                          (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'une conversion est invalide.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BADCASTEXCEPTION_H
#define ARCCORE_BASE_BADCASTEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Exception lorsqu'une conversion d'un type vers un autre est invalide.
 */
class ARCCORE_BASE_EXPORT BadCastException
: public Exception
{
 public:

  explicit BadCastException(const String& where);
  BadCastException(const String& where, const String& message);
  explicit BadCastException(const TraceInfo& where);
  BadCastException(const TraceInfo& where, const String& message);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
