// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDispatchers.h                                              (C) 2000-2025 */
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

namespace Arcane::MessagePassing
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
  virtual ITypeDispatcher<BFloat16>* dispatcher(BFloat16*) = 0;
  virtual ITypeDispatcher<Float16>* dispatcher(Float16*) = 0;

  virtual IControlDispatcher* controlDispatcher() = 0;
  virtual ISerializeDispatcher* serializeDispatcher() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
