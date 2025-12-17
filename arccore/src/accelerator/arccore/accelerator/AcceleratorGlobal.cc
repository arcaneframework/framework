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
#include "arccore/accelerator/Atomic.h"
#include "arccore/accelerator/LocalMemory.h"
#include "arccore/accelerator/Reduce.h"
#include "arccore/accelerator/GenericFilterer.h"
#include "arccore/accelerator/GenericPartitioner.h"
#include "arccore/accelerator/GenericReducer.h"
#include "arccore/accelerator/GenericScanner.h"
#include "arccore/accelerator/GenericSorter.h"
#include "arccore/accelerator/SpanViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file RunCommandLoop.h
 *
 * \brief Types et macros pour gérer les boucles sur les accélérateurs
 */

/*!
 * \file Reduce.h
 *
 * \brief Types et fonctions pour gérer les synchronisations sur les accélérateurs
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::String Arcane::Accelerator::impl::
getBadPolicyMessage(eExecutionPolicy policy)
{
  switch (policy) {
  case eExecutionPolicy::CUDA:
    return "Requesting CUDA kernel execution but the kernel is not compiled with CUDA."
           " You need to compile the file containing this kernel with CUDA compiler.";
  case eExecutionPolicy::HIP:
    return "Requesting HIP kernel execution but the kernel is not compiled with HIP."
           " You need to compile the file containing this kernel with HIP compiler.";
  case eExecutionPolicy::SYCL:
    return "Requesting SYCL kernel execution but the kernel is not compiled with SYCL."
           " You need to compile the file containing this kernel with SYCL compiler.";
  default:
    break;
  }
  return String::format("Invalid execution policy '{0}'", policy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
