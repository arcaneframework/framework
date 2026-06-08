// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexOutOfRangeException.h                                  (C) 2000-2025 */
/*                                                                           */
/* Exception when a value is not within a given range.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INDEXOUTOFRANGEEXCEPTION_H
#define ARCCORE_BASE_INDEXOUTOFRANGEEXCEPTION_H
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
 * \brief Exception when a value is not within a given range.
 *
 * Indicates that minValue() <= index() < maxValue() is not respected.
 */
class ARCCORE_BASE_EXPORT IndexOutOfRangeException
: public Exception
{
 public:

  IndexOutOfRangeException(const TraceInfo& where, const String& message,
                           Int64 index, Int64 min_value_inclusive,
                           Int64 max_value_exclusive);

 public:

  void explain(std::ostream& m) const override;

  //! Index
  Int64 index() const { return m_index; }
  //! Valid minimum value (inclusive)
  Int64 minValue() const { return m_min_value_inclusive; }
  //! Valid maximum value (exclusive)
  Int64 maxValue() const { return m_max_value_exclusive; }

 private:

  Int64 m_index;
  Int64 m_min_value_inclusive;
  Int64 m_max_value_exclusive;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
