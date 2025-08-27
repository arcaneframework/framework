// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadOperationException.h                                     (C) 2000-2025 */
/*                                                                           */
/* Exception sur une opération des expressions.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_EXPR_BADOPERATIONEXCEPTION_H
#define ARCANE_CORE_EXPR_BADOPERATIONEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

#include "arcane/core/datatype/VariantBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exception sur une opération des expressions.
 *
 * Cette exception est envoyée lorsqu'un script essaie d'utiliser une 
 * opération non définie sur des expressions.
 */
class ARCANE_EXPR_EXPORT BadOperationException
: public Exception
{
 public:
  
  BadOperationException(const String& where,const String& operationName,
                        VariantBase::eType operandType);
  BadOperationException(const BadOperationException& ex);
  ~BadOperationException() ARCANE_NOEXCEPT{}

 public:

  virtual void explain(std::ostream& m) const;

 private:

  String m_operation_name;
  VariantBase::eType m_operand_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

