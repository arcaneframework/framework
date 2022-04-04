// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InvalidArgumentException.h                                  (C) 2000-2016 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INVALIDARGUMENTEXCEPTION_H
#define ARCANE_UTILS_INVALIDARGUMENTEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exception lorsqu'une erreur fatale est survenue.
 */
class ARCANE_UTILS_EXPORT InvalidArgumentException
: public Exception
{
 public:

  InvalidArgumentException(const TraceInfo& where,const String& arg_name,int arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,double arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const String& arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const void* arg);

  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const String& message,int arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const String& message,double arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const String& message,const String& arg);
  InvalidArgumentException(const TraceInfo& where,const String& arg_name,const String& message,const void* arg);

  InvalidArgumentException(const InvalidArgumentException& ex);

  ~InvalidArgumentException() ARCANE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:
  
  String m_arg_name;
  String m_arg_value;
  String m_message;

  template<typename U>
  void _init(const U& arg_value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
