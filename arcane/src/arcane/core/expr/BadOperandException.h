// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadOperandException.h                                       (C) 2000-2018 */
/*                                                                           */
/* Exception sur les opérandes des opérations des expressions.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_BADOPERANDEXCEPTION_H
#define ARCANE_EXPR_BADOPERANDEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exception sur les opérandes des opérations des expressions.
 *
 * Cette exception est envoyée lorsque les opérandes des opérations des 
 * des expressions n'ont pas le bon type ou la bonne dimension.
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

