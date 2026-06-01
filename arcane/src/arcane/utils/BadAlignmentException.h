// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadAlignmentException.h                                     (C) 2000-2017 */
/*                                                                           */
/* Exception when an address is not correctly aligned.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_BADCASTEXCEPTION_H
#define ARCANE_UTILS_BADCASTEXCEPTION_H
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
 * \ingroup Core
 * \brief Exception when an address is not correctly aligned.
 */
class ARCANE_UTILS_EXPORT BadAlignmentException
: public Exception
{
 public:

  BadAlignmentException(const String& where, const void* ptr, Integer alignment);
  BadAlignmentException(const TraceInfo& where, const void* ptr, Integer alignment);

  virtual void explain(std::ostream& m) const;

 private:

  const void* m_ptr;
  Integer m_wanted_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
