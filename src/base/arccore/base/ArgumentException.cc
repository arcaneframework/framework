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
/* ArgumentException.cc                                        (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/ArgumentException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const String& awhere)
: Exception("ArgumentException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const String& awhere,const String& amessage)
: Exception("ArgumentException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const TraceInfo& awhere)
: Exception("ArgumentException",awhere,"Bad argument")
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const TraceInfo& awhere,const String& amessage)
: Exception("ArgumentException",awhere,amessage)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
ArgumentException(const ArgumentException& rhs) ARCCORE_NOEXCEPT
: Exception(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArgumentException::
~ArgumentException() ARCCORE_NOEXCEPT
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

