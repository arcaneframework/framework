// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReduceMemoryImpl.h                                          (C) 2000-2023 */
/*                                                                           */
/* Gestion de la mémoire pour les réductions.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_REDUCEMEMORYIMPL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_REDUCEMEMROYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/IReduceMemoryImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReduceMemoryImpl
: public IReduceMemoryImpl
{
 public:

  ReduceMemoryImpl(RunCommandImpl* p)
  : m_command(p)
  , m_device_memory_bytes(eMemoryRessource::Device)
  , m_host_memory_bytes(eMemoryRessource::HostPinned)
  , m_grid_buffer(eMemoryRessource::Device)
  , m_grid_device_count(eMemoryRessource::Device)
  {
    _allocateMemoryForReduceData(128);
    _allocateMemoryForGridDeviceCount();
  }

 public:

  void* allocateReduceDataMemory(ConstMemoryView identity_view) override;
  void setGridSizeAndAllocate(Int32 grid_size) override
  {
    m_grid_size = grid_size;
    _setReducePolicy();
    _allocateGridDataMemory();
  }
  Int32 gridSize() const override { return m_grid_size; }

  GridMemoryInfo gridMemoryInfo() override
  {
    return m_grid_memory_info;
  }
  void copyReduceValueFromDevice() override;
  void release() override;

 private:

  RunCommandImpl* m_command = nullptr;

  //! Pointeur vers la mémoire unifiée contenant la donnée réduite
  std::byte* m_device_memory = nullptr;

  //! Allocation pour la donnée réduite en mémoire managée
  NumArray<std::byte, MDDim1> m_device_memory_bytes;

  //! Allocation pour la donnée réduite en mémoire hôte
  NumArray<std::byte, MDDim1> m_host_memory_bytes;

  //! Taille allouée pour \a m_device_memory
  Int64 m_size = 0;

  //! Taille courante de la grille (nombre de blocs)
  Int32 m_grid_size = 0;

  //! Taille de la donnée actuelle
  Int64 m_data_type_size = 0;

  GridMemoryInfo m_grid_memory_info;

  //! Tableau contenant la valeur de la réduction pour chaque bloc d'une grille
  NumArray<Byte, MDDim1> m_grid_buffer;

  //! Buffer pour conserver la valeur de l'identité
  UniqueArray<std::byte> m_identity_buffer;

  /*!
   * \brief Tableau de 1 entier non signé contenant le nombre de grilles ayant déja
   * effectuée la réduction.
   */
  NumArray<unsigned int, MDDim1> m_grid_device_count;

 private:

  void _allocateGridDataMemory();
  void _allocateMemoryForGridDeviceCount();
  void _setReducePolicy();
  void _allocateMemoryForReduceData(Int32 new_size)
  {
    m_device_memory_bytes.resize(new_size);
    m_device_memory = m_device_memory_bytes.to1DSpan().data();

    m_host_memory_bytes.resize(new_size);
    m_grid_memory_info.m_host_memory_for_reduced_value = m_host_memory_bytes.to1DSpan().data();

    m_size = new_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
