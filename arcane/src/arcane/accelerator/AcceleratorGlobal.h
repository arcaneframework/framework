// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorGlobal.h                                         (C) 2000-2021 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ACCELERATORGLOBAL_H
#define ARCANE_ACCELERATOR_ACCELERATORGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_accelerator
#define ARCANE_ACCELERATOR_EXPORT ARCANE_EXPORT
#else
#define ARCANE_ACCELERATOR_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Initialise \a runner en fonction de
 * la valeur de \a acc_info.
 */
extern "C++" ARCANE_ACCELERATOR_EXPORT void
initializeRunner(Runner& runner, ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour indiquer qu'un noyau n'a pas été compilé avec HIP
#define ARCANE_FATAL_NO_HIP_COMPILATION() \
  ARCANE_FATAL("Requesting HIP kernel execution but the kernel is not compiled with HIP." \
               " You need to compile the file containing this kernel with HIP compiler.")

//! Macro pour indiquer qu'un noyau n'a pas été compilé avec CUDA
#define ARCANE_FATAL_NO_CUDA_COMPILATION() \
  ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA." \
               " You need to compile the file containing this kernel with CUDA compiler.")

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
