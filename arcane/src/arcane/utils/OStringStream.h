// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OStringStream.h                                             (C) 2000-2018 */
/*                                                                           */
/* Flux de sortie dans une chaîne de caractères.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_OSTRINGSTREAM_H
#define ARCANE_UTILS_OSTRINGSTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OStringStreamPrivate;

/*!
 * \brief Flot de sortie lié à une String.
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

  OStringStreamPrivate* m_p; //!< Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
