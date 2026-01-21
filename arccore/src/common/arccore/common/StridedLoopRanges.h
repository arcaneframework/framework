// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StridedLoopRanges.h                                         (C) 2000-2026 */
/*                                                                           */
/* Gestion du lancement des noyaux de calcul sur accélérateur.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_STRIDEDLOOPRANGES_H
#define ARCCORE_COMMON_STRIDEDLOOPRANGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour gérer la décomposition d'une boucle en plusieurs parties.
 */
class ARCCORE_COMMON_EXPORT StridedLoopRangesBase
{
 public:

  constexpr StridedLoopRangesBase(Int32 nb_stride, Int64 nb_orig_element)
  : m_stride_value((nb_orig_element + (nb_stride - 1)) / nb_stride)
  , m_nb_original_element(nb_orig_element)
  , m_nb_stride(nb_stride)
  {
  }
  constexpr StridedLoopRangesBase(Int64 nb_orig_element)
  : m_stride_value(nb_orig_element)
  , m_nb_original_element(nb_orig_element)
  , m_nb_stride(1)
  {
  }

 public:

  constexpr Int32 nbStride() const { return m_nb_stride; }
  constexpr Int64 nbOriginalElement() const { return m_nb_original_element; }
  constexpr Int64 strideValue() const { return m_stride_value; }

  void setNbStride(Int32 nb_stride) { _setNbStride(nb_stride); }

 private:

  //! Valeur du pas
  Int64 m_stride_value = 0;
  //! Nombre d'éléments dans la boucle d'origine
  Int64 m_nb_original_element = 0;
  //! Nombre de pas
  Int32 m_nb_stride = 0;

 private:

  void _setNbStride(Int32 nb_stride)
  {
    m_nb_stride = nb_stride;
    m_stride_value = (m_nb_original_element + (nb_stride - 1)) / nb_stride;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour gérer la décomposition d'une boucle en plusieurs parties.
 */
template <typename LoopBoundType_>
class StridedLoopRanges
: public StridedLoopRangesBase
{
 public:

  using LoopBoundType = LoopBoundType_;

 public:

  StridedLoopRanges(Int32 nb_grid_stride, const LoopBoundType& orig_loop)
  : StridedLoopRangesBase(nb_grid_stride, orig_loop.nbElement())
  , m_orig_loop(orig_loop)
  {
  }
  StridedLoopRanges(const LoopBoundType& orig_loop)
  : StridedLoopRangesBase(orig_loop.nbElement())
  , m_orig_loop(orig_loop)
  {
  }
  constexpr const LoopBoundType& originalLoop() const { return m_orig_loop; }

 private:

  LoopBoundType m_orig_loop;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LoopBoundType> ARCCORE_HOST_DEVICE LoopBoundType::LoopIndexType
arcaneGetLoopIndexCudaHip(StridedLoopRanges<LoopBoundType> loop_bounds, Int32 index)
{
  return arcaneGetLoopIndexCudaHip(loop_bounds.originalLoop(), index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
