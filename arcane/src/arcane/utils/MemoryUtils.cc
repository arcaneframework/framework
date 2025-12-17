// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryUtils.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/String.h"
#include "arccore/common/internal/IMemoryResourceMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MemoryUtils.h
 * \brief Fonctions de gestion mémoire et des allocateurs.
 */
/*!
 * \namespace Arcane::MemoryUtils
 * \brief Espace de noms pour les fonctions de gestion mémoire et des allocateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MemoryUtils::impl::
computeCapacity(Int64 size)
{
  double d_size = static_cast<double>(size);
  double d_new_capacity = d_size * 1.8;
  if (size > 5000000)
    d_new_capacity = d_size * 1.2;
  else if (size > 500000)
    d_new_capacity = d_size * 1.5;
  return static_cast<Int64>(d_new_capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
