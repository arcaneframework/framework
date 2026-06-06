// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadVariantTypeException.cc                                  (C) 2000-2018 */
/*                                                                           */
/* Exception raised when a variant is not of the desired type                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/Iostream.h"

#include "arcane/datatype/BadVariantTypeException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadVariantTypeException::
BadVariantTypeException(const String& where, VariantBase::eType wrongType)
: Exception("BadVariantType", where)
, m_wrong_type(wrongType)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadVariantTypeException::
explain(std::ostream& m) const
{
  m << "Invalid type for a variant: " << VariantBase::typeName(m_wrong_type)
    << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
