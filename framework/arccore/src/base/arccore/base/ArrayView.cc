// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/TraceInfo.h"

// On n'utilise pas directement ces fichiers mais on les inclus pour tester
// la compilation. Lorsque les tests seront en place on pourra supprmer
// ces inclusions
#include "arccore/base/Array2View.h"
#include "arccore/base/Array3View.h"
#include "arccore/base/Array4View.h"
#include "arccore/base/Span.h"
#include "arccore/base/Span2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
