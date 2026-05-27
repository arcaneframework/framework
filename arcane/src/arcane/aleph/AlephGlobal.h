// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephGlobal.h                                               (C) 2000-2024 */
/*                                                                           */
/* General declarations for the 'arcane_aleph' component.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_ALEPHGLOBAL_H
#define ARCANE_ALEPH_ALEPHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_aleph
#define ARCANE_ALEPH_EXPORT ARCANE_EXPORT
#else
#define ARCANE_ALEPH_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAlephFactory;
class IAlephTopology;
class IAlephMatrix;
class IAlephVector;
class AlephKernel;
class AlephTopology;
class AlephMatrix;
class AlephOrdering;
class AlephIndexing;
class AlephVector;
class AlephParams;

//! Default type for indexing rows and columns of matrices and vectors
using AlephInt = int;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
