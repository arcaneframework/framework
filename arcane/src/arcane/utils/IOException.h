// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IOException.h                                               (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une erreur d'entrée/sortie est détectée.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IOEXCEPTION_H
#define ARCANE_UTILS_IOEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Exception lorsqu'une erreur d'entrée/sortie est détectée.
 */
class ARCANE_UTILS_EXPORT IOException
: public Exception
{
 public:
	
  IOException(const String& where);
  IOException(const String& where,const String& message);
  IOException(const TraceInfo& where);
  IOException(const TraceInfo& where,const String& message);
  IOException(const IOException& ex);
  ~IOException() ARCANE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:

	String m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

