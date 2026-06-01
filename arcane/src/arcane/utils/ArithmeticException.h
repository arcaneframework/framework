// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArithmeticException.h                                       (C) 2000-2014 */
/*                                                                           */
/* Exception when an arithmetic error occurs.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARITHMETICEXCEPTION_H
#define ARCANE_UTILS_ARITHMETICEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Exception when an arithmetic error occurs.
 *
 * This exception occurs notably when a SIGFPE signal occurs
 */
class ARCANE_UTILS_EXPORT ArithmeticException
: public Exception
{
 public:

  ArithmeticException(const TraceInfo& where);
  ArithmeticException(const TraceInfo& where, const String& message);
  ArithmeticException(const TraceInfo& where, const StackTrace& stack_trace);
  ArithmeticException(const TraceInfo& where, const String& message,
                      const StackTrace& stack_trace);
  ArithmeticException(const ArithmeticException& ex)
  : Exception(ex)
  {}
  ~ArithmeticException() ARCANE_NOEXCEPT {}

 public:

  ArithmeticException() = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
