// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadOperandException.h                                       (C) 2000-2018 */
/*                                                                           */
/* Exception for operands in expression operations.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_BADOPERANDEXCEPTION_H
#define ARCANE_EXPR_BADOPERANDEXCEPTION_H
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
 * \brief Exception for operands in expression operations.
 *
 * This exception is thrown when the operands of the
 * expressions do not have the correct type or dimension.
 */
class BadOperandException
: public Exception
{
 public:

  BadOperandException(const String& where);

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
