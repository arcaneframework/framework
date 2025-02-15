// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeoutException.h                                          (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'un signal survient.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_TIMEOUTEXCEPTION_H
#define ARCCORE_BASE_TIMEOUTEXCEPTION_H
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
 * \internal
 * \brief Exception lorsqu'un timeout survient.
 */
class ARCCORE_BASE_EXPORT TimeoutException
: public Exception
{
 public:

  TimeoutException(const String& where);
  TimeoutException(const String& where,const StackTrace& stack_trace);
  TimeoutException(const TimeoutException& ex) : Exception(ex){}
  ~TimeoutException() ARCCORE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

