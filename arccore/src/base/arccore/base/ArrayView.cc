// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/FatalErrorException.h"

// On n'utilise pas directement ces fichiers mais on les inclus pour tester
// la compilation. Lorsque les tests seront en place on pourra supprimer
// ces inclusions
#include "arccore/base/Array2View.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"
#include "arccore/base/Span.h"
#include "arccore/base/Span2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file Span.h
 * \brief Types et fonctions associés aux classes SpanImpl, SmallSpan and Span.
 */
/*!
 * \file Span2.h
 * \brief Types et fonctions associés aux classes Span2Impl, Small2Span and Span2.
 */
/*!
 * \file ArrayView.h
 * \brief Types et fonctions associés aux classes ArrayView et ConstArrayView.
 */
/*!
 * \file Array2View.h
 * \brief Types et fonctions associés aux classes Array2View et ConstArray2View.
 */
/*!
 * \file Array3View.h
 * \brief Types et fonctions associés aux classes Array3View et ConstArray3View.
 */
/*!
 * \file Array4View.h
 * \brief Types et fonctions associés aux classes Array4View et ConstArray4View.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Lance une exception 'ArgumentException'
ARCCORE_BASE_EXPORT void impl::
arccoreThrowTooBigInteger [[noreturn]] (std::size_t size)
{
  ARCCORE_THROW(ArgumentException,"value '{0}' too big for Array size",size);
}

//! Lance une exception 'ArgumentException'
ARCCORE_BASE_EXPORT void impl::
arccoreThrowTooBigInt64 [[noreturn]] (std::size_t size)
{
  ARCCORE_THROW(ArgumentException,"value '{0}' too big to fit in Int64",size);
}

ARCCORE_BASE_EXPORT void impl::
arccoreThrowNegativeSize [[noreturn]] (Int64 size)
{
  ARCCORE_THROW(ArgumentException,"invalid negative value '{0}' for Array size",size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
binaryWrite(std::ostream& ostr,const Span<const std::byte>& bytes)
{
  auto* ptr = reinterpret_cast<const char*>(bytes.data());
  ostr.write(ptr,bytes.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
binaryRead(std::istream& istr,const Span<std::byte>& bytes)
{
  auto* ptr = reinterpret_cast<char*>(bytes.data());
  istr.read(ptr,bytes.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Impl::ExtentStorageBase::
_throwBadSize(Int64 wanted_size, Int64 expected_size)
{
  ARCCORE_FATAL("Bad size value for fixed extent size={0} expected={1}", wanted_size, expected_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
