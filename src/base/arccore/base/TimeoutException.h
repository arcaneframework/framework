// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeoutException.h                                          (C) 2000-2018 */
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

namespace Arccore
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

