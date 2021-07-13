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
/* IDispatchers.h                                              (C) 2000-2018 */
/*                                                                           */
/* Interface du conteneur des dispatchers.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IDISPATCHERS_H
#define ARCCORE_MESSAGEPASSING_IDISPATCHERS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
class IControlDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du conteneur des dispatchers.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IDispatchers
{
 public:

  virtual ~IDispatchers() = default;

 public:

  virtual ITypeDispatcher<char>* dispatcher(char*) = 0;
  virtual ITypeDispatcher<signed char>* dispatcher(signed char*) = 0;
  virtual ITypeDispatcher<unsigned char>* dispatcher(unsigned char*) = 0;
  virtual ITypeDispatcher<short>* dispatcher(short*) = 0;
  virtual ITypeDispatcher<unsigned short>* dispatcher(unsigned short*) = 0;
  virtual ITypeDispatcher<int>* dispatcher(int*) = 0;
  virtual ITypeDispatcher<unsigned int>* dispatcher(unsigned int*) = 0;
  virtual ITypeDispatcher<long>* dispatcher(long*) = 0;
  virtual ITypeDispatcher<unsigned long>* dispatcher(unsigned long*) = 0;
  virtual ITypeDispatcher<long long>* dispatcher(long long*) = 0;
  virtual ITypeDispatcher<unsigned long long>* dispatcher(unsigned long long*) = 0;
  virtual ITypeDispatcher<float>* dispatcher(float*) = 0;
  virtual ITypeDispatcher<double>* dispatcher(double*) = 0;
  virtual ITypeDispatcher<long double>* dispatcher(long double*) = 0;

  virtual IControlDispatcher* controlDispatcher() = 0;
  virtual ISerializeDispatcher* serializeDispatcher() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
