// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayMetaData.h                                             (C) 2000-2026 */
/*                                                                           */
/* Tableau 1D.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAYMETADATA_H
#define ARCCORE_COMMON_ARRAYMETADATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryAllocationOptions.h"
#include "arccore/common/MemoryAllocationArgs.h"
#include "arccore/common/IMemoryAllocator.h"
#include "arccore/common/AllocatedMemoryInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Meta-Données des tableaux.
 *
 * Cette classe sert pour contenir les meta-données communes à toutes les
 * implémentations qui dérivent de AbstractArray.
 *
 * Seules les classes qui implémentent une sémantique à la UniqueArray
 * ont le droit d'utiliser un allocateur autre que l'allocateur par défaut.
 */
class ARCCORE_COMMON_EXPORT ArrayMetaData
{
  // NOTE: Les champs de cette classe sont utilisés pour l'affichage TTF de totalview.
  // Si on modifie leur ordre il faut mettre à jour la copie de cette classe
  // dans l'afficheur totalview de Arcane.

  template <typename> friend class AbstractArray;
  template <typename> friend class Array2;
  template <typename> friend class Array;
  template <typename> friend class SharedArray;
  template <typename> friend class SharedArray2;
  friend class AbstractArrayBase;
  static IMemoryAllocator* _defaultAllocator();

 public:

  ArrayMetaData()
  : allocation_options(_defaultAllocator())
  {}

 protected:

  //! Nombre d'éléments du tableau (pour les tableaux 1D)
  Int64 size = 0;
  //! Taille de la première dimension (pour les tableaux 2D)
  Int64 dim1_size = 0;
  //! Taille de la deuxième dimension (pour les tableaux 2D)
  Int64 dim2_size = 0;
  //! Nombre d'éléments alloués
  Int64 capacity = 0;
  //! Allocateur mémoire et options associées
  MemoryAllocationOptions allocation_options;
  //! Nombre de références sur l'instance
  Int32 nb_ref = 0;
  //! Indique is cette instance a été allouée par l'opérateur new.
  bool is_allocated_by_new = false;
  //! Indique si cette instance n'est pas l'instance nulle (partagée par tous les SharedArray)
  bool is_not_null = false;

 protected:

  IMemoryAllocator* _allocator() const { return allocation_options.m_allocator; }

 public:

  static void throwInvalidMetaDataForSharedArray ARCCORE_NORETURN();
  static void throwNullExpected ARCCORE_NORETURN();
  static void throwNotNullExpected ARCCORE_NORETURN();
  static void throwUnsupportedSpecificAllocator ARCCORE_NORETURN();
  static void overlapError ARCCORE_NORETURN(const void* begin1, Int64 size1,
                                            const void* begin2, Int64 size2);

 protected:

  using MemoryPointer = void*;
  using ConstMemoryPointer = const void*;

 protected:

  MemoryPointer _allocate(Int64 nb, Int64 sizeof_true_type, RunQueue* queue);
  MemoryPointer _reallocate(const AllocatedMemoryInfo& mem_info, Int64 new_capacity, Int64 sizeof_true_type, RunQueue* queue);
  void _deallocate(const AllocatedMemoryInfo& mem_info, RunQueue* queue) noexcept
  {
    if (_allocator()) {
      MemoryAllocationArgs alloc_args = _getAllocationArgs(queue);
      _allocator()->deallocate(alloc_args, mem_info);
    }
  }
  MemoryPointer _changeAllocator(const MemoryAllocationOptions& new_allocator_opt, const AllocatedMemoryInfo& current_info, Int64 sizeof_true_type, RunQueue* queue);
  void _setMemoryLocationHint(eMemoryLocationHint new_hint, void* ptr, Int64 sizeof_true_type);
  void _setHostDeviceMemoryLocation(eHostDeviceMemoryLocation location);
  void _copyFromMemory(MemoryPointer destination, ConstMemoryPointer source, Int64 sizeof_true_type, RunQueue* queue);

 private:

  void _checkAllocator() const;
  MemoryAllocationArgs _getAllocationArgs() const { return allocation_options.allocationArgs(); }
  MemoryAllocationArgs _getAllocationArgs(RunQueue* queue) const
  {
    return allocation_options.allocationArgs(queue);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Ce type n'est plus utilisé.
 */
class ArrayImplBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Cette classe n'est plus utilisée.
 */
template <typename T>
class ArrayImplT
: public ArrayImplBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
