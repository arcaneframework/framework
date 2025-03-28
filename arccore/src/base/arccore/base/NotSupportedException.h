// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NotSupportedException.h                                     (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'une opération n'est pas supportée.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_NOTSUPPORTEDEXCEPTION_H
#define ARCCORE_BASE_NOTSUPPORTEDEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception lorsqu'une opération n'est pas supportée.
 */
class ARCCORE_BASE_EXPORT NotSupportedException
: public Exception
{
 public:
	
  explicit NotSupportedException(const String& where);
  NotSupportedException(const String& where,const String& message);
  explicit NotSupportedException(const TraceInfo& where);
  NotSupportedException(const TraceInfo& where,const String& message);
  NotSupportedException(const NotSupportedException& ex) ARCCORE_NOEXCEPT;
  ~NotSupportedException() ARCCORE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:

  String m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

