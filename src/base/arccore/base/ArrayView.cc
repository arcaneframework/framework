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
/* ArrayView.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/TraceInfo.h"

// On n'utilise pas directement ces fichiers mais on les inclus pour tester
// la compilation. Lorsque les tests seront en place on pourra supprmer
// ces inclusions
#include "arccore/base/Array2View.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"
#include "arccore/base/Span.h"
#include "arccore/base/Span2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(long size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(unsigned int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Integer
arccoreCheckArraySize(int size)
{
  if (size>=ARCCORE_INTEGER_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
  if (size<0)
    ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
  return (Integer)size;
}

extern "C++" ARCCORE_BASE_EXPORT Int64
arccoreCheckLargeArraySize(size_t size)
{
  if (size>=ARCCORE_INT64_MAX)
    ARCCORE_THROW(ArgumentException,"value '{0}' too big to fit in Int64",size);
  return (Int64)size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
