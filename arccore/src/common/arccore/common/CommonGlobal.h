// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonGlobal.h                                              (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Common' de 'Arccore'.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_COMMONGLOBAL_H
#define ARCCORE_COMMON_COMMONGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPONENT_arccore_common)
#define ARCCORE_COMMON_EXPORT ARCCORE_EXPORT
#define ARCCORE_COMMON_EXTERN_TPL
#else
#define ARCCORE_COMMON_EXPORT ARCCORE_IMPORT
#define ARCCORE_COMMON_EXTERN_TPL extern
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

