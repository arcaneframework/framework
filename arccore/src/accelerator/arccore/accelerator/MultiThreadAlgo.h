// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiThreadAlgo.h                                           (C) 2000-2026 */
/*                                                                           */
/* Implementation of accelerator algorithms in multi-thread mode.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_MULTITHREADALGO_H
#define ARCCORE_ACCELERATOR_MULTITHREADALGO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/SmallArray.h"

#include "arccore/base/ForLoopRunInfo.h"
#include "arccore/concurrency/ParallelFor.h"

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Advanced algorithms in multi-thread mode.
 *
 * Currently, only the Scan operation is implemented.
 */
class MultiThreadAlgo
{
 public:

  /*!
   * \brief Multi-thread scan algorithm.
   *
   * \note This class is internal to Arcane. The public API version
   * is accessible via the GenericScanner class.
   *
   * This basic algorithm uses two passes for calculation.
   * The iteration interval is divided into N blocks. We take N = 2*nb_thread.
   * - the first pass calculates the scan result in parallel for all
   * elements of a block.
   * - the second pass calculates the final value.
   *
   * The calculation always yields the same value for a given number of blocks.
   *
   * TODO: Use padding to maintain partial values per block.
   * TODO: Create specialized versions if DataType is a base type
   * such as 'Int32', 'Int64', 'float', or 'double'.
   */
  template <bool IsExclusive, typename DataType, typename Operator,
            typename InputIterator, typename OutputIterator>
  void doScan(ForLoopRunInfo run_info, Int32 nb_value,
              InputIterator input, OutputIterator output,
              DataType init_value, Operator op)
  {
    //std::cout << "DO_SCAN MULTI_THREAD nb_value=" << nb_value << " init_value=" << init_value << "\n";
    auto multiple_getter_func = [=](Int32 input_index, Int32 nb_value) -> DataType {
      DataType partial_value = Operator::defaultValue();
      for (Int32 x = 0; x < nb_value; ++x)
        partial_value = op(input[x + input_index], partial_value);
      return partial_value;
    };

    auto multiple_setter_func = [=](DataType previous_sum, Int32 input_index, Int32 nb_value) {
      for (Int32 x = 0; x < nb_value; ++x) {
        if constexpr (IsExclusive) {
          output[x + input_index] = previous_sum;
          previous_sum = op(input[x + input_index], previous_sum);
        }
        else {
          previous_sum = op(input[x + input_index], previous_sum);
          output[x + input_index] = previous_sum;
        }
      }
    };
    // TODO: automatically calculate this value.
    const Int32 nb_block = 10;

    // Array to store partial values of the blocks.
    // TODO: Use padding to avoid cache conflicts between threads.
    SmallArray<DataType> partial_values(nb_block);
    Span<DataType> out_partial_values = partial_values;

    auto partial_value_func = [=](Int32 a, Int32 n) {
      for (Int32 i = 0; i < n; ++i) {
        Int32 interval_index = i + a;

        Int32 input_index = 0;
        Int32 nb_value_in_interval = 0;
        _subInterval<Int32>(nb_value, interval_index, nb_block, &input_index, &nb_value_in_interval);

        DataType partial_value = multiple_getter_func(input_index, nb_value_in_interval);

        out_partial_values[interval_index] = partial_value;
      }
    };

    ParallelLoopOptions loop_options(run_info.options().value_or(ParallelLoopOptions{}));
    loop_options.setGrainSize(1);
    run_info.addOptions(loop_options);

    // Calculates partial sums for nb_block
    Arcane::arccoreParallelFor(0, nb_block, run_info, partial_value_func);

    auto final_sum_func = [=](Int32 a, Int32 n) {
      for (Int32 i = 0; i < n; ++i) {
        Int32 interval_index = i + a;

        DataType previous_sum = init_value;
        for (Int32 z = 0; z < interval_index; ++z)
          previous_sum = op(out_partial_values[z], previous_sum);

        Int32 input_index = 0;
        Int32 nb_value_in_interval = 0;
        _subInterval<Int32>(nb_value, interval_index, nb_block, &input_index, &nb_value_in_interval);

        multiple_setter_func(previous_sum, input_index, nb_value_in_interval);
      }
    };

    // Calculates the final values
    Arcane::arccoreParallelFor(0, nb_block, run_info, final_sum_func);
  }

  template <bool InPlace, typename InputIterator, typename OutputIterator, typename SelectLambda>
  Int32 doFilter(ForLoopRunInfo run_info, Int32 nb_value,
                 InputIterator input, OutputIterator output,
                 SelectLambda select_lambda)
  {
    // Index type.
    using IndexType = Int32;

    UniqueArray<bool> select_flags(nb_value);
    Span<bool> select_flags_view = select_flags;
    //std::cout << "DO_FILTER MULTI_THREAD nb_value=" << nb_value << "\n";
    auto multiple_getter_func = [=](Int32 input_index, Int32 nb_value) -> IndexType {
      IndexType partial_value = 0;
      for (Int32 x = 0; x < nb_value; ++x) {
        const Int32 index = x + input_index;
        bool is_select = select_lambda(input[index]);
        select_flags_view[index] = is_select;
        if (is_select)
          ++partial_value;
      }
      return partial_value;
    };

    auto multiple_setter_func = [=](IndexType partial_value, Int32 input_index, Int32 nb_value) {
      for (Int32 x = 0; x < nb_value; ++x) {
        const Int32 index = x + input_index;
        if (select_flags_view[index]) {
          output[partial_value] = input[index];
          ++partial_value;
        }
      }
    };

    // TODO: automatically calculate this value.
    const Int32 nb_block = 10;

    // Array to store partial values of the blocks.
    // TODO: Use padding to avoid cache conflicts between threads.
    SmallArray<Int32> partial_values(nb_block, 0);
    Span<Int32> out_partial_values = partial_values;

    auto partial_value_func = [=](Int32 a, Int32 n) {
      for (Int32 i = 0; i < n; ++i) {
        Int32 interval_index = i + a;

        Int32 input_index = 0;
        Int32 nb_value_in_interval = 0;
        _subInterval<Int32>(nb_value, interval_index, nb_block, &input_index, &nb_value_in_interval);

        out_partial_values[interval_index] = multiple_getter_func(input_index, nb_value_in_interval);
      }
    };

    ParallelLoopOptions loop_options(run_info.options().value_or(ParallelLoopOptions{}));
    loop_options.setGrainSize(1);
    run_info.addOptions(loop_options);

    // Calculates partial sums for nb_block
    Arcane::arccoreParallelFor(0, nb_block, run_info, partial_value_func);

    // Calculates the number of filtered values
    // Also calculates the accumulated value of partial_values
    Int32 nb_filter = 0;
    for (Int32 i = 0; i < nb_block; ++i) {
      Int32 x = partial_values[i];
      nb_filter += x;
      partial_values[i] = nb_filter;
    }

    auto filter_func = [=](Int32 a, Int32 n) {
      for (Int32 i = 0; i < n; ++i) {
        Int32 interval_index = i + a;

        IndexType partial_value = 0;
        if (interval_index > 0)
          partial_value = out_partial_values[interval_index - 1];

        Int32 input_index = 0;
        Int32 nb_value_in_interval = 0;
        _subInterval<Int32>(nb_value, interval_index, nb_block, &input_index, &nb_value_in_interval);

        multiple_setter_func(partial_value, input_index, nb_value_in_interval);
      }
    };

    // If the input and output are the same, the filling is done sequentially.
    // TODO: do it in parallel.
    if (InPlace)
      filter_func(0, nb_block);
    else
      Arcane::arccoreParallelFor(0, nb_block, run_info, filter_func);

    return nb_filter;
  }

 private:

  template <typename SizeType>
  static void _subInterval(SizeType size, SizeType interval_index, SizeType nb_interval,
                           SizeType* out_begin_index, SizeType* out_interval_size)
  {
    *out_begin_index = 0;
    *out_interval_size = 0;
    if (nb_interval <= 0)
      return;
    if (interval_index < 0 || interval_index >= nb_interval)
      return;
    SizeType isize = size / nb_interval;
    SizeType ibegin = interval_index * isize;
    // For the last interval, take the remaining elements
    if ((interval_index + 1) == nb_interval)
      isize = size - ibegin;
    *out_begin_index = ibegin;
    *out_interval_size = isize;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
