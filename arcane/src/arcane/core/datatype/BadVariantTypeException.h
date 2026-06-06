// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadVariantTypeException.h                                   (C) 2000-2025 */
/*                                                                           */
/* Exception raised when a variant is not of the desired type                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPE_BADVARIANTTYPEEXCEPTION_H
#define ARCANE_CORE_DATATYPE_BADVARIANTTYPEEXCEPTION_H
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
 * \internal
 *
 * \brief Exception for an invalid variant type.
 *
 * This exception is thrown when trying to construct a variant
 * with an unknown type.
 */
class BadVariantTypeException
: public Exception
{
 public:

  BadVariantTypeException(const String& where, VariantBase::eType wrongType);

 public:

  virtual void explain(std::ostream& m) const;

 private:

  VariantBase::eType m_wrong_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
