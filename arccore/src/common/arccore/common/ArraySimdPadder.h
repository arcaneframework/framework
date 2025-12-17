// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArraySimdPadder.h                                           (C) 2000-2025 */
/*                                                                           */
/* Classe pour ajouter du 'padding' pour la vectorisation.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAYSIMDPADDER_H
#define ARCCORE_COMMON_ARRAYSIMDPADDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"
#include "arccore/common/Array.h"
#include "arccore/common/AlignedMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArraySimdPadder
{
 public:

  /*!
   * \brief Calcule la taille nécessaire pour être un multiple de \a PaddingSize.
   *
   * \a SizeType peut être un Int32 ou un Int64
   */
  template <int PaddingSize, typename SizeType> ARCCORE_HOST_DEVICE inline static SizeType
  getSizeWithSpecificPadding(SizeType size)
  {
    if (size <= 0)
      return 0;
    SizeType modulo = size % PaddingSize;
    if (modulo == 0)
      return size;
    // TODO: vérifier débordement.
    SizeType padding_size = ((size / PaddingSize) + 1) * PaddingSize;
    return padding_size;
  }

  /*!
   * \brief Calcule la taille nécessaire pour être un multiple de SIMD_PADDING_SIZE.
   *
   * \a SizeType peut être un Int32 ou un Int64
   */
  template <typename SizeType> ARCCORE_HOST_DEVICE inline static SizeType
  getSizeWithPadding(SizeType size)
  {
    return getSizeWithSpecificPadding<SIMD_PADDING_SIZE>(size);
  }

  template <typename DataType>
  static bool isNeedPadding(Span<const DataType> ids)
  {
    using SizeType = Int64;
    const SizeType size = ids.size();
    SizeType padding_size = getSizeWithPadding(size);
    return (padding_size > size);
  }

  template <typename DataType> ARCCORE_HOST_DEVICE static void applySimdPaddingView(Span<DataType> ids)
  {
    using SizeType = Int64;
    const SizeType size = ids.size();
    SizeType padding_size = getSizeWithPadding(size);
    if (padding_size <= size)
      return;

    // Construit une vue avec la taille du padding
    // pour éviter les débordement de tableau lors des tests
    Span<DataType> padded_ids(ids.data(), padding_size);

    DataType last_value = ids[size - 1];
    for (SizeType k = size; k < padding_size; ++k)
      padded_ids[k] = last_value;
  }

  template <typename DataType>
  static void applySimdPadding(Array<DataType>& ids)
  {
    const Int64 size = ids.largeSize();
    Int64 padding_size = getSizeWithPadding(size);
    if (padding_size <= size)
      return;
    MemoryAllocationArgs args;
    if (ids.allocator()->guaranteedAlignment(args) < AlignedMemoryAllocator::simdAlignment())
      ARCCORE_FATAL("Allocator guaranted alignment ({0}) has to be greated than {1}",
                   ids.allocator()->guaranteedAlignment(args), AlignedMemoryAllocator::simdAlignment());
    if (padding_size > ids.capacity())
      ARCCORE_FATAL("Not enough capacity c={0} min_expected={1}", ids.capacity(),
                   padding_size);
    applySimdPaddingView(ids.span());
  }

  template <typename DataType>
  static void checkSimdPadding(Span<const DataType> ids)
  {
    using SizeType = Int64;
    const Int64 size = ids.size();
    SizeType padding_size = getSizeWithPadding(size);
    if (padding_size <= size)
      return;

    // Construit une vue avec la taille du padding
    // pour éviter les débordements de tableau lors des tests
    Span<const DataType> padded_ids(ids.data(), padding_size);

    // Vérifie que le padding est fait avec la dernière valeur valide.
    SizeType last_id = ids[size - 1];
    for (SizeType k = size; k < padding_size; ++k)
      if (padded_ids[k] != last_id)
        ARCCORE_FATAL("Bad padding value i={0} expected={1} value={2}",
                     k, last_id, padded_ids[k]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
