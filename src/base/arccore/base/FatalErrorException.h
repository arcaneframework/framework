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
/* FatalErrorException.h                                       (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une erreur fatale est survenue.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FATALERROREXCEPTION_H
#define ARCCORE_BASE_FATALERROREXCEPTION_H
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
 * \ingroup Core
 * \brief Exception lorsqu'une erreur fatale est survenue.
 */
class ARCCORE_BASE_EXPORT FatalErrorException
: public Exception
{
 public:
	
  explicit FatalErrorException(const String& where);
  explicit FatalErrorException(const TraceInfo& where);
  FatalErrorException(const String& where,const String& message);
  FatalErrorException(const TraceInfo& where,const String& message);
  FatalErrorException(const FatalErrorException& rhs) ARCCORE_NOEXCEPT;
  ~FatalErrorException() ARCCORE_NOEXCEPT {}

 public:
	
  void explain(std::ostream& m) const override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
