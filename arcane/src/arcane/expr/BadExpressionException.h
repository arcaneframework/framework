// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadExpressionException.h                                    (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une expression n'est pas valide.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EXPR_BADEXPRESSIONEXCEPTION_H
#define ARCANE_EXPR_BADEXPRESSIONEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exception lorsqu'une expression n'est pas valide.
 */
class BadExpressionException
: public Exception
{
 public:
	
  BadExpressionException(const String& where);
  BadExpressionException(const String& where,const String& msg);
  ~BadExpressionException() ARCANE_NOEXCEPT override {}

 public:
	
  void explain(std::ostream& m) const override;

 private:

  String m_msg;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

