// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArgumentException.h                                         (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARGUMENTEXCEPTION_H
#define ARCCORE_BASE_ARGUMENTEXCEPTION_H
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
 * \ingroup Core
 * \brief Exception lorsqu'un argument est invalide.
 */
class ARCCORE_BASE_EXPORT ArgumentException
: public Exception
{
 public:
	
  explicit ArgumentException(const String& where);
  ArgumentException(const String& where,const String& message);
  explicit ArgumentException(const TraceInfo& where);
  ArgumentException(const TraceInfo& where,const String& message);
  ArgumentException(const ArgumentException& rhs) ARCCORE_NOEXCEPT;
  ~ArgumentException() ARCCORE_NOEXCEPT override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arrcore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

