// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLaunchArgs.h                                          (C) 2000-2025 */
/*                                                                           */
/* Arguments pour lancer un kernel.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_KERNELLAUNCHARGS_H
#define ARCCORE_COMMON_ACCELERATOR_KERNELLAUNCHARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Arguments pour lancer un kernel.
 */
class ARCCORE_COMMON_EXPORT KernelLaunchArgs
{
  friend RunCommandLaunchInfo;

 public:

  KernelLaunchArgs() = default;
  KernelLaunchArgs(Int32 nb_block_per_grid, Int32 nb_thread_per_block)
  : m_nb_block_per_grid(nb_block_per_grid)
  , m_nb_thread_per_block(nb_thread_per_block)
  {
  }

 public:

  int nbBlockPerGrid() const { return m_nb_block_per_grid; }
  int nbThreadPerBlock() const { return m_nb_thread_per_block; }

 private:

  int m_nb_block_per_grid = 0;
  int m_nb_thread_per_block = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
