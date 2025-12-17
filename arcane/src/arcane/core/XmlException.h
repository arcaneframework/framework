// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlException.h                                              (C) 2000-2025 */
/*                                                                           */
/* Exception sur les opérandes des opérations des expressions.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_XMLEXCEPTION_H
#define ARCANE_CORE_XMLEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  XmlException(const String& where, const String& msg)
  : Exception("XmlException", where)
  {
    setMessage(msg);
  }
  XmlException(const TraceInfo& where, const String& msg)
  : Exception("XmlException", where)
  {
    setMessage(msg);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
