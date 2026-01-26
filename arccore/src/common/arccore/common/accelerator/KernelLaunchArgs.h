// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KernelLaunchArgs.h                                          (C) 2000-2026 */
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

  //! Nombre de blocs de la grille
  Int32 nbBlockPerGrid() const { return m_nb_block_per_grid; }
  //! Nombre de blocs de la grille
  void setNbBlockPerGrid(Int32 v) { m_nb_block_per_grid = v; }

  //! Nombre de threads par bloc
  Int32 nbThreadPerBlock() const { return m_nb_thread_per_block; }
  //! Nombre de threads par bloc
  void setNbThreadPerBlock(Int32 v) { m_nb_thread_per_block = v; }

  //! Mémoire partagée à allouer pour le noyau
  Int32 sharedMemorySize() const { return m_shared_memory_size; }
  //! Mémoire partagée à allouer pour le noyau
  void setSharedMemorySize(Int32 v) { m_shared_memory_size = v; }

  //! Indique si on lance en mode coopératif (i.e. cudaLaunchCooperativeKernel)
  bool isCooperative() const { return m_is_cooperative; }
  //! Indique si on lance en mode coopératif
  void setIsCooperative(bool v) { m_is_cooperative = v; }

 private:

  Int32 m_nb_block_per_grid = 0;
  Int32 m_nb_thread_per_block = 0;
  Int32 m_shared_memory_size = 0;
  bool m_is_cooperative = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
