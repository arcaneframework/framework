// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IReduceMemoryImpl.h                                         (C) 2000-2026 */
/*                                                                           */
/* Interface de la gestion mémoire pour les réductions.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_IREDUCEMEMORYIMPL_H
#define ARCCORE_COMMON_ACCELERATOR_IREDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la gestion mémoire pour les réductions.
 */
class ARCCORE_COMMON_EXPORT IReduceMemoryImpl
{
 public:

  //! Informations mémoire pour la réduction sur les accélérateurs
  struct GridMemoryInfo
  {
    //! Mémoire allouée pour la réduction sur une grille (de taille nb_bloc * sizeof(T))
    MutableMemoryView m_grid_memory_values;
    //! Entier utilisé pour compter le nombre de blocs ayant déjà fait leur partie de la réduction
    unsigned int* m_grid_device_count = nullptr;
    /*!
     * \brief Pointeur vers la mémoire sur l'hôte contenant la valeur réduite.
     *
     * Cette mémoire est punaisée et est donc accessible depuis l'accélérateur.
     */
    void* m_host_memory_for_reduced_value = nullptr;
  };

 public:

  virtual ~IReduceMemoryImpl() = default;

 public:

  /*!
   * \brief Alloue la mémoire pour une donnée dont on veut faire une réduction.
   *
   * \a data_type_size est la taille de la donnée.
   */
  virtual void allocateReduceDataMemory(Int32 data_type_size) = 0;

  //! Positionne la taille de la grille GPU (le nombre de blocs)
  virtual void setGridSizeAndAllocate(Int32 grid_size) = 0;

  //! Taille de la grille GPU (nombre de blocs)
  virtual Int32 gridSize() const = 0;

  //! Informations sur la mémoire utilisée par la réduction
  virtual GridMemoryInfo gridMemoryInfo() = 0;

  //! Libère l'instance.
  virtual void release() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
