// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IReduceMemoryImpl.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface de la gestion mémoire pour les réductions.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_IREDUCEMEMORYIMPL_H
#define ARCANE_ACCELERATOR_CORE_IREDUCEMEMORYIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

#include "arcane/utils/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la gestion mémoire pour les réductions.
 * \warning API en cours de définition.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl
{
 public:

  //! Informations mémoire pour la réduction sur les accélérateurs
  struct GridMemoryInfo
  {
    //! Mémoire allouée pour la réduction sur une grille (de taille nb_bloc * sizeof(T))
    MutableMemoryView m_grid_memory_values;
    //! Entier utilisé pour compter le nombre de blocs ayant déjà fait leur partie de la réduction
    unsigned int* m_grid_device_count = nullptr;
    //! Politique de réduction
    eDeviceReducePolicy m_reduce_policy = eDeviceReducePolicy::Grid;
    //! Pointeur vers la mémoire sur l'hôte contenant la valeur réduite.
    void* m_host_memory_for_reduced_value = nullptr;
    //! Taille d'un warp
    Int32 m_warp_size = 64;
  };

 public:

  virtual ~IReduceMemoryImpl() = default;

 public:

  /*!
   * \brief Alloue la mémoire pour une donnée dont on veut faire une réduction et
   * remplit la zone avec la valeur de \a identity_view.
   */
  virtual void* allocateReduceDataMemory(ConstMemoryView identity_view) = 0;

  //! Positionne la taille de la grille GPU (le nombre de blocs)
  virtual void setGridSizeAndAllocate(Int32 grid_size) = 0;

  //! Taille de la grille GPU (nombre de blocs)
  virtual Int32 gridSize() const = 0;

  //! Informations sur la mémoire utilisée par la réduction
  virtual GridMemoryInfo gridMemoryInfo() = 0;

  /*!
   * \brief Copie la valeur réduite depuis le device vers l'hote.
   *
   * La valeur sera copié de gridMemoryInfo().m_device_memory_for_reduced_value
   * vers gridMemoryInfo().m_host_memory_for_reduced_value
   */
  virtual void copyReduceValueFromDevice() =0;

  //! Libère l'instance.
  virtual void release() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Alloue la mémoire pour contenir la valeur réduite et positionne
 * sa valeur à \a identity.
 */
template<typename T> T*
allocateReduceDataMemory(IReduceMemoryImpl* p,T identity)
{
  T* ptr = reinterpret_cast<T*>(p->allocateReduceDataMemory(makeMemoryView(&identity)));
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
