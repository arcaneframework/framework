// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OStringStream.h                                             (C) 2000-2018 */
/*                                                                           */
/* Output stream into a character string.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_OSTRINGSTREAM_H
#define ARCANE_UTILS_OSTRINGSTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OStringStreamPrivate;

/*!
 * \brief Output stream linked to a String.
 */
class ARCANE_UTILS_EXPORT OStringStream
{
 public:

  OStringStream();
  explicit OStringStream(Integer bufsize);
  ~OStringStream();

  OStringStream(const OStringStream& rhs) = delete;
  void operator=(const OStringStream& rhs) = delete;

 public:

  std::ostream& operator()();
  std::ostream& stream();
  String str();
  void reset();

 private:

  OStringStreamPrivate* m_p; //!< Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
