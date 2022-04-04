// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArithmeticException.h                                       (C) 2000-2014 */
/*                                                                           */
/* Exception lorsqu'une erreur arithmétique survient.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARITHMETICEXCEPTION_H
#define ARCANE_UTILS_ARITHMETICEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Exception lorsqu'une erreur arithmétique survient.
 *
 * Cette exception survient notamment lorsqu'un signal SIGFPE survient
 */
class ARCANE_UTILS_EXPORT ArithmeticException
: public Exception
{
 public:
	
  ArithmeticException(const TraceInfo& where);
  ArithmeticException(const TraceInfo& where,const String& message);
  ArithmeticException(const TraceInfo& where,const StackTrace& stack_trace);
  ArithmeticException(const TraceInfo& where,const String& message,
                      const StackTrace& stack_trace);
  ArithmeticException(const ArithmeticException& ex) : Exception(ex){}
  ~ArithmeticException() ARCANE_NOEXCEPT {}

 public:

  ArithmeticException() = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

