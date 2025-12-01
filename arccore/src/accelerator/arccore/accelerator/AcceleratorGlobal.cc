// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorGlobal.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

// Les fichiers suivants servent à tester que tout compile bien
#include "arccore/accelerator/LocalMemory.h"
#include "arccore/accelerator/Reduce.h"
#include "arccore/accelerator/GenericFilterer.h"
#include "arccore/accelerator/GenericPartitioner.h"
#include "arccore/accelerator/GenericReducer.h"
#include "arccore/accelerator/GenericScanner.h"
#include "arccore/accelerator/GenericSorter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
