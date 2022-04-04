// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlException.h                                              (C) 2000-2009 */
/*                                                                           */
/* Exception sur les opérandes des opérations des expressions.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_XMLEXCEPTION_H
#define ARCANE_XMLEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Xml
 * \brief Exception liées aux fichiers XML.
 */
class XmlException
: public Exception
{
 public:
  
  XmlException(const String& where,const String& msg)
  : Exception("XmlException",where)
  {
    setMessage(msg);
  }
  XmlException(const TraceInfo& where,const String& msg)
  : Exception("XmlException",where)
  {
    setMessage(msg);
  }
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

