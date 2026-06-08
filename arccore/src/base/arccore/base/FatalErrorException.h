// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FatalErrorException.h                                       (C) 2000-2025 */
/*                                                                           */
/* Exception when a fatal error has occurred.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FATALERROREXCEPTION_H
#define ARCCORE_BASE_FATALERROREXCEPTION_H
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
 * \brief Exception when a fatal error has occurred.
 */
class ARCCORE_BASE_EXPORT FatalErrorException
: public Exception
{
 public:

  explicit FatalErrorException(const String& where);
  explicit FatalErrorException(const TraceInfo& where);
  FatalErrorException(const String& where, const String& message);
  FatalErrorException(const TraceInfo& where, const String& message);
  FatalErrorException(const FatalErrorException& rhs) ARCCORE_NOEXCEPT;
  ~FatalErrorException() ARCCORE_NOEXCEPT {}

 public:

  void explain(std::ostream& m) const override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
