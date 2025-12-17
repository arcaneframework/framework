// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScanImpl.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Implémentation spécifique de l'opération de scan pour les accélérateurs.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_SCANIMPL_H
#define ARCCORE_ACCELERATOR_SCANIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/FatalErrorException.h"

#include "arccore/common/NumArray.h"
#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/accelerator/RunCommandLaunchInfo.h"

#include "arccore/accelerator/CommonUtils.h"
#include "arccore/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

/*!
 * \brief Algorithme de Scan basique.
 *
 * \note Avec Intel DPC++ 2024.1, l'opérateur de Scan doit être un des opérateurs
 * sycl de base (comme sycl::plus) pour qu'on puisse utiliser
 * sycl::inclusive_scan_over_group() avec. Ce n'est pas le cas avec AdaptiveCpp
 * 24.02 qui n'a pas cette limitation.
 */
template <bool IsExclusive, typename DataType, typename Operator>
class SyclScanner
{
  class InputInfo
  {
   public:

    DataType _getInputValue(Int32 index) const
    {
      DataType local_value = identity;
      if constexpr (IsExclusive) {
        if (index == 0)
          local_value = init_value;
        else
          local_value = ((index - 1) < nb_value) ? input_values[index - 1] : identity;
      }
      else
        local_value = (index < nb_value) ? input_values[index] : identity;
      return local_value;
    }

   public:

    SmallSpan<const DataType> input_values;
    DataType identity = {};
    DataType init_value = {};
    Int32 nb_value = 0;
  };

 public:

  void doScan(RunQueue& rq, SmallSpan<const DataType> input, SmallSpan<DataType> output, DataType init_value)
  {
    DataType identity = Operator::defaultValue();
    sycl::queue q = Impl::SyclUtils::toNativeStream(&rq);
    // Contient l'application partielle de Operator pour chaque bloc de thread
    NumArray<DataType, MDDim1> tmp;
    // Contient l'application partielle de Operator cumulée avec les blocs précédents
    NumArray<DataType, MDDim1> tmp2;
    Int32 nb_item = input.size();
    Int32 block_size = 256;
    Int32 nb_block = (nb_item / block_size);
    if ((nb_item % block_size) != 0)
      ++nb_block;
    tmp.resize(nb_block);
    tmp2.resize(nb_block);
    InputInfo input_info;
    input_info.nb_value = nb_item;
    input_info.init_value = init_value;
    input_info.identity = identity;
    input_info.input_values = input;
    if (m_is_verbose)
      std::cout << "DO_SCAN nb_item=" << nb_item << " nb_block=" << nb_block << "\n";
    doscan1(q, input_info, tmp.to1DSpan(), nb_item, block_size);
    if (m_is_verbose)
      for (int i = 0; i < nb_block; ++i)
        std::cout << "DO_SCAN_X1 i=" << i << " tmp[i]=" << tmp[i] << "\n";
    doscan2(q, tmp.to1DSpan(), nb_block, block_size, identity);
    if (m_is_verbose)
      for (int i = 0; i < nb_block; ++i)
        std::cout << "DO_SCAN_X2 i=" << i << " tmp[i]=" << tmp[i] << "\n";
    doscan2_bis(q, tmp.to1DSpan(), tmp2.to1DSpan(), nb_block, block_size, identity);
    if (m_is_verbose)
      for (int i = 0; i < nb_block; ++i)
        std::cout << "DO_SCAN_X2_BIS i=" << i << " tmp[i]=" << tmp[i] << " tmp2[i]=" << tmp2[i] << "\n";
    doscan3(q, input_info, output, tmp2, nb_item, block_size);
  }

 private:

  void doscan1(sycl::queue& q, const InputInfo& input_info, Span<DataType> tmp,
               int nb_value, int block_size)
  {
    if (m_is_verbose)
      std::cout << "DO_SCAN1 nb_value=" << nb_value << " L=" << block_size << "\n";
    // Phase 1: Compute local scans over input blocks
    Operator scan_op;
    q.submit([&](sycl::handler& h) {
       auto local = sycl::local_accessor<DataType, 1>(block_size, h);
       h.parallel_for(_getNDRange(nb_value, block_size), [=](sycl::nd_item<1> it) {
         const int i = static_cast<int>(it.get_global_id(0));
         const int li = static_cast<int>(it.get_local_id(0));
         const int gid = static_cast<int>(it.get_group(0));
         const int local_range0 = static_cast<int>(it.get_local_range()[0]);
         // Effectue le scan sur le groupe.
         DataType local_value = input_info._getInputValue(i);
         local[li] = sycl::inclusive_scan_over_group(it.get_group(), local_value, scan_op.syclFunctor());
         // Le dernier élément sauve la valeur dans le tableau du groupe.
         if (li == local_range0 - 1)
           tmp[gid] = local[li];
       });
     })
    .wait();
  }

  void doscan2(sycl::queue& q, Span<DataType> tmp, int nb_block, const int block_size, DataType identity)
  {
    if (m_is_verbose)
      std::cout << "DO_SCAN2 nb_block=" << nb_block << " block_size=" << block_size << "\n";
    // Phase 2: Compute scan over partial results
    Operator scan_op;
    q.submit([&](sycl::handler& h) {
       auto local = sycl::local_accessor<DataType, 1>(block_size, h);
       h.parallel_for(_getNDRange(nb_block, block_size), [=](sycl::nd_item<1> it) {
         int i = static_cast<int>(it.get_global_id(0));
         int li = static_cast<int>(it.get_local_id(0));
         // Copy input to local memory
         DataType local_value = (i < nb_block) ? tmp[i] : identity;
         local[li] = sycl::inclusive_scan_over_group(it.get_group(), local_value, scan_op.syclFunctor());
         // Overwrite result from each work-item in the temporary buffer
         if (i < nb_block)
           tmp[i] = local[li];
       });
     })
    .wait();
  }

  void doscan2_bis(sycl::queue& q, Span<const DataType> tmp, Span<DataType> tmp2, int nb_block, int block_size, DataType identity)
  {
    if (m_is_verbose)
      std::cout << "DO_SCAN2_bis nb_block=" << nb_block << " L=" << block_size << "\n";
    Operator scan_op;
    q.parallel_for(_getNDRange(nb_block, block_size), [=](sycl::nd_item<1> it) {
       const int g = static_cast<int>(it.get_group(0));
       const int i = static_cast<int>(it.get_global_id(0));
       if (i < nb_block) {
         DataType init_value = identity;
         for (int j = 1; j <= g; ++j)
           init_value = scan_op(init_value, tmp[(j * block_size) - 1]);
         tmp2[i] = scan_op(init_value, tmp[i]);
       }
     })
    .wait();
  }

  void doscan3(sycl::queue& q, const InputInfo& input_info, SmallSpan<DataType> output, SmallSpan<DataType> tmp2, int nb_value, int block_size)
  {
    if (m_is_verbose)
      std::cout << "DO_SCAN3 nb_value=" << nb_value << " L=" << block_size << "\n";
    Operator scan_op;
    // Phase 3: Update local scans using partial results
    q.parallel_for(_getNDRange(nb_value, block_size), [=](sycl::nd_item<1> it) {
       const int i = static_cast<int>(it.get_global_id(0));
       const int g = static_cast<int>(it.get_group(0));
       DataType local_value = input_info._getInputValue(i);
       DataType output_value = sycl::inclusive_scan_over_group(it.get_group(), local_value, scan_op.syclFunctor());
       if (i < nb_value) {
         output[i] = (g > 0) ? scan_op(output_value, tmp2[g - 1]) : output_value;
       }
     })
    .wait();
  }

 private:

  bool m_is_verbose = false;

 private:

  //! Return le premier multiple de \a block_size supérieur ou égal à \a nb_value
  sycl::nd_range<1> _getNDRange(Int32 nb_value, Int32 block_size)
  {
    int x = nb_value / block_size;
    if ((nb_value % block_size) != 0)
      ++x;
    x *= block_size;
    return sycl::nd_range<1>(x, block_size);
  }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
