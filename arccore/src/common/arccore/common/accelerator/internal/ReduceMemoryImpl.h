// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReduceMemoryImpl.h                                          (C) 2000-2025 */
/*                                                                           */
/* Gestion de la mémoire pour les réductions.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_REDUCEMEMORYIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_REDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/IReduceMemoryImpl.h"

#include "arccore/common/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReduceMemoryImpl
: public IReduceMemoryImpl
{
 public:

  explicit ReduceMemoryImpl(RunCommandImpl* p);

 public:

  void allocateReduceDataMemory(Int32 data_type_size) override;
  void setGridSizeAndAllocate(Int32 grid_size) override
  {
    m_grid_size = grid_size;
    _allocateGridDataMemory();
  }
  Int32 gridSize() const override { return m_grid_size; }

  GridMemoryInfo gridMemoryInfo() override
  {
    return m_grid_memory_info;
  }
  void release() override;

 private:

  RunCommandImpl* m_command = nullptr;

  //! Allocation pour la donnée réduite en mémoire hôte
  UniqueArray<std::byte> m_host_memory_bytes;

  //! Taille allouée pour \a m_device_memory
  Int64 m_size = 0;

  //! Taille courante de la grille (nombre de blocs)
  Int32 m_grid_size = 0;

  //! Taille de la donnée actuelle
  Int64 m_data_type_size = 0;

  GridMemoryInfo m_grid_memory_info;

  //! Tableau contenant la valeur de la réduction pour chaque bloc d'une grille
  UniqueArray<Byte> m_grid_buffer;

  //! Buffer pour conserver la valeur de l'identité
  UniqueArray<std::byte> m_identity_buffer;

  /*!
   * \brief Tableau de 1 entier non signé contenant le nombre de grilles ayant déja
   * effectuée la réduction.
   */
  UniqueArray<unsigned int> m_grid_device_count;

 private:

  void _allocateGridDataMemory();
  void _allocateMemoryForGridDeviceCount();
  void _setReducePolicy();
  void _allocateMemoryForReduceData(Int32 new_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
