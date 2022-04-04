// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadCastException.h                                          (C) 2000-2014 */
/*                                                                           */
/* Exception lorsqu'une conversion est invalide.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_BADCASTEXCEPTION_H
#define ARCANE_UTILS_BADCASTEXCEPTION_H
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
 * \brief Exception lorsqu'une conversion d'un type vers un autre est invalide.
 */
class ARCANE_UTILS_EXPORT BadCastException
: public Exception
{
 public:
	
  explicit BadCastException(const String& where);
  BadCastException(const String& where,const String& message);
  explicit BadCastException(const TraceInfo& where);
  BadCastException(const TraceInfo& where,const String& message);
  BadCastException(const BadCastException& rhs) ARCANE_NOEXCEPT : Exception(rhs){}
  ~BadCastException() ARCANE_NOEXCEPT {}

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

