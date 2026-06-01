// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelFatalErrorException.h                               (C) 2000-2018 */
/*                                                                           */
/* Exception when a 'parallel' fatal error occurred.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PARALLELFATALERROREXCEPTION_H
#define ARCANE_UTILS_PARALLELFATALERROREXCEPTION_H
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
 * \ingroup Parallel
 * \brief Exception when a 'parallel' fatal error is generated.
 *
 * A 'parallel' fatal error is a fatal error common to all
 * subdomains. In this case, it is possible to cleanly stop the code
 */
class ARCANE_UTILS_EXPORT ParallelFatalErrorException
: public Exception
{
 public:

  ParallelFatalErrorException(const String& where);
  ParallelFatalErrorException(const TraceInfo& where);
  ParallelFatalErrorException(const String& where, const String& message);
  ParallelFatalErrorException(const TraceInfo& where, const String& message);
  ParallelFatalErrorException(const ParallelFatalErrorException& rhs)
  : Exception(rhs)
  {}
  ~ParallelFatalErrorException() ARCANE_NOEXCEPT {}

 public:

  virtual void explain(std::ostream& m) const;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
