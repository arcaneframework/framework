// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephGlobal.h                                               (C) 2000-2022 */
/*                                                                           */
/* Déclarations générales de la composante 'arcane_aleph'.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_ALEPHGLOBAL_H
#define ARCANE_ALEPH_ALEPHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

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
class AlephTopology;
class AlephMatrix;
class AlephOrdering;
class AlephIndexing;
class AlephVector;
class IParallelMng;
class ISubDomain;
class AlephParams;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
