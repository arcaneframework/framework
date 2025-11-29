// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Test.sycl.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Fichier contenant les tests pour l'implémentation SYCL.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/sycl/SyclAccelerator.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Scan.h"

#include "arcane/utils/NumArray.h"

using namespace Arccore;
using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Test Appel pure SYCL
extern "C" int arcaneTestSycl1()
{
  const int N = 8;
  std::cout << "TEST1\n";

  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int* data = sycl::malloc_shared<int>(N, q);

  for (int i = 0; i < N; i++)
    data[i] = i;

  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
     data[i] *= 2;
   })
  .wait();

  for (int i = 0; i < N; i++)
    std::cout << data[i] << std::endl;
  sycl::free(data, q);

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Idem Test1 avec des NumArray
extern "C" int arcaneTestSycl2()
{
  const int N = 8;
  std::cout << "TEST 2\n";

  sycl::queue q;

  NumArray<Int32, MDDim1> data(N);

  for (int i = 0; i < N; i++)
    data[i] = i;

  Span<Int32> inout_data(data.to1DSpan());
  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
     inout_data[i] *= 3;
   })
  .wait();

  for (int i = 0; i < N; i++)
    std::cout << data[i] << std::endl;

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Idem Test1 avec des NumArray
extern "C" int arcaneTestSycl3()
{
  const int N = 12;
  std::cout << "TEST 3\n";

  Runner runner_sycl(eExecutionPolicy::SYCL);
  RunQueue queue{makeQueue(runner_sycl)};
  sycl::queue q;

  NumArray<Int32, MDDim1> data(N);

  for (int i = 0; i < N; i++)
    data[i] = i;

  {
    auto command = makeCommand(queue);
    Span<Int32> inout_data(data.to1DSpan());
    command << RUNCOMMAND_LOOP1(iter, N)
    {
      auto [i] = iter();
      inout_data[i] *= 4;
    };
  }

  for (int i = 0; i < N; i++)
    std::cout << data[i] << std::endl;

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" int arcaneTestSycl4()
{
  // device.get_info<cl::sycl::info::device::max_work_group_size>();
  //constexpr Int32 WARP_SIZE = 32;
  constexpr Int32 BLOCK_SIZE = 128;

  //const int nb_block = 152 * 15 * 12;
  const int NB_BLOCK = 152;
  const int N = BLOCK_SIZE * NB_BLOCK;
  std::cout << "TEST 4\n";

  sycl::device device{ sycl::gpu_selector_v };
  Int64 mcu = device.get_info<sycl::info::device::max_compute_units>();
  Int64 mwg = device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "DEVICE mcu=" << mcu << " mwg=" << mwg << "\n";
  sycl::queue q{ device };

  NumArray<Int32, MDDim2> data1(N, 6);
  NumArray<Int32, MDDim1> data_to_reduce(N);
  NumArray<Int64, MDDim1> data_partial_reduce(NB_BLOCK);
  NumArray<Int32, MDDim1> atomic_counter(eMemoryRessource::Device);
  atomic_counter.resize(1);

  Int64 ref_total_reduce = 0;
  for (int i = 0; i < N; i++) {
    data_to_reduce[i] = i;
    ref_total_reduce += data_to_reduce[i];
  }

  Span<Int32> out_atomic_counter(atomic_counter.to1DSpan());
  {
    q.single_task([=]() {
      out_atomic_counter[0] = 0;
    });
  }

  const int nb_iter = 1;
  for (Int32 iter = 0; iter < nb_iter; ++iter) {
    MDSpan<Int32, MDDim2> inout_data1(data1.mdspan());
    Span<Int32> in_data_to_reduce(data_to_reduce.to1DSpan());
    Span<Int64> inout_data_partial_reduce(data_partial_reduce.to1DSpan());
    Int32* atomic_counter_ptr = out_atomic_counter.data();
    q.parallel_for(sycl::nd_range<1>(N, BLOCK_SIZE), [=](sycl::nd_item<1> id) {
       Int32 i = static_cast<Int32>(id.get_global_id());
       const Int32 global_id = static_cast<Int32>(id.get_global_id(0));
       const Int32 local_id = static_cast<Int32>(id.get_local_id(0));
       const Int32 group_id = static_cast<Int32>(id.get_group_linear_id());
       const Int32 sub_group_id = static_cast<Int32>(id.get_sub_group().get_local_id());
       Int32 nb_block = static_cast<Int32>(id.get_group_range(0));
       //Int32 nb_thread = static_cast<Int32>(id.get_local_range(0));
       inout_data1(i, 0) = global_id;
       inout_data1(i, 1) = local_id;
       inout_data1(i, 2) = group_id;
       inout_data1(i, 3) = sub_group_id;
       inout_data1(i, 5) = 0;
       Int32 v = in_data_to_reduce[i];
       Int32 local_sum = 0;
       bool is_last = false;
       id.barrier(sycl::access::fence_space::local_space);
       //Int32 v2_bis =  id.get_sub_group().shuffle_down(v,1);
       Int32 vx = sycl::reduce_over_group(id.get_group(),v,sycl::plus<Int32>{});
       inout_data1(i, 0) = vx;
       if (local_id == 0) {
         //Int32 base = global_id;
         //for (Int32 x = 0; x < nb_thread; ++x)
         //local_sum += in_data_to_reduce[x + base];
         local_sum = vx;
         inout_data1(i, 4) = local_sum;
         inout_data_partial_reduce[group_id] = local_sum;
         sycl::atomic_ref<Int32, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*atomic_counter_ptr);
         Int32 cx = a.fetch_add(1);
         inout_data1(i, 5) = cx;
         if (cx == (nb_block - 1))
           is_last = true;
       }
       id.barrier(sycl::access::fence_space::local_space);
       // Je suis le dernier à faire la réduction.
       // Calcule la réduction finale
       if (is_last) {
         Int64 my_total = 0;
         for (int x = 0; x < nb_block; ++x)
           my_total += inout_data_partial_reduce[x];
         // Met le résultat final dans le premier élément du tableau.
         inout_data_partial_reduce[0] = my_total;
         *atomic_counter_ptr = 0;
       }
     })
    .wait();
  }
  Int64 kernel_total = data_partial_reduce[0];
  std::cout << "N=" << N << " REF_TOTAL=" << ref_total_reduce << " computed=" << kernel_total << "\n";
  bool do_verbose = true;
  if (do_verbose) {
    for (int i = 0; i < N; i++) {
      Int32 imod = i % 32;
      if (imod < 2)
        std::cout << "I=" << i << " global_id=" << data1(i, 0)
                  << " local_id=" << data1(i, 1)
                  << " group_id=" << data1(i, 2)
                  << " sub_group_local_id=" << data1(i, 3)
                  << " v=" << data1(i, 4)
                  << std::endl;
    }
  }
  std::cout << "FINAL_N=" << N << " REF_TOTAL=" << ref_total_reduce << " computed=" << kernel_total << "\n";
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" void arcaneTestSycl5()
{
  Runner runner(eExecutionPolicy::SYCL);
  RunQueue queue{ makeQueue(runner) };
  constexpr int N = 25;

  NumArray<Int32, MDDim1> data(N);

  for (int i = 0; i < N; i++)
    data[i] = i;

  {
    auto command = makeCommand(queue);
    Span<Int32> inout_data(data.to1DSpan());
    ReducerSum<Int64> reducer1(command);
    command << RUNCOMMAND_LOOP1(iter, N)
    {
      auto [i] = iter();
      reducer1.add(inout_data[i]);
      inout_data[i] *= 4;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" void arcaneTestSycl6()
{
  Runner runner(eExecutionPolicy::SYCL);
  RunQueue queue{ makeQueue(runner) };
  //constexpr int N = 63;
  //constexpr int N = 139;
  //constexpr int N = 256;
  constexpr int N = 4789;
  // A TESTER
  //constexpr int N = 16900;
  //constexpr int N = 1000000;
  NumArray<Int64, MDDim1> data(N);
  NumArray<Int64, MDDim1> out_data(N);
  NumArray<Int64, MDDim1> expected_inclusive_data(N);
  NumArray<Int64, MDDim1> expected_exclusive_data(N);

  Int64 total = 0;
  Int64 total_exclusive = 7;
  for (int i = 0; i < N; i++) {
    expected_exclusive_data[i] = total_exclusive;
    data[i] = (i + 2);
    total += data[i];
    total_exclusive += data[i];
    expected_inclusive_data[i] = total;
  }

  Arcane::Accelerator::impl::SyclScanner<false, Int64, ScannerSumOperator<Int64>> scanner;
  scanner.doScan(queue, data.to1DSmallSpan(), out_data.to1DSmallSpan(), 7);

  const bool do_verbose = (N < 256);
  for (int i = 0; i < N; i++) {
    bool is_bad = out_data[i] != expected_inclusive_data[i];
    if (do_verbose || is_bad)
      std::cout << "OUT_INCL=" << i << " v=" << out_data[i] << " expected=" << expected_inclusive_data[i] << "\n";
    if (is_bad)
      ARCANE_FATAL("Bad value");
  }
  std::cout << "FINAL OUT_INCL=" << (N - 1) << " v=" << out_data[N - 1] << " expected=" << expected_inclusive_data[N - 1] << "\n";

  Arcane::Accelerator::impl::SyclScanner<true, Int64, ScannerSumOperator<Int64>> scanner2;
  scanner2.doScan(queue, data.to1DSmallSpan(), out_data.to1DSmallSpan(), 7);

  for (int i = 0; i < N; i++) {
    bool is_bad = out_data[i] != expected_exclusive_data[i];
    if (do_verbose || is_bad)
      std::cout << "OUT_EXCL=" << i << " v=" << out_data[i] << " expected=" << expected_exclusive_data[i] << "\n";
    if (is_bad)
      ARCANE_FATAL("Bad value");
  }
  std::cout << "FINAL OUT_EXCL=" << (N - 1) << " v=" << out_data[N - 1] << " expected=" << expected_exclusive_data[N - 1] << "\n";
}

extern "C" void arcaneTestSycl7()
{
  Runner runner(eExecutionPolicy::SYCL);
  RunQueue queue{ makeQueue(runner) };
  constexpr int N = 63;
  //constexpr int N = 139;
  //constexpr int N = 256;
  //constexpr int N = 4789;
  // A TESTER
  //constexpr int N = 16900;
  //constexpr int N = 1000000;
  NumArray<Int64, MDDim1> data(N);
  NumArray<Int64, MDDim1> out_data(N);
  NumArray<Int64, MDDim1> expected_inclusive_data(N);
  NumArray<Int64, MDDim1> expected_exclusive_data(N);

  Int64 total = 0;
  Int64 total_exclusive = 7;
  for (int i = 0; i < N; i++) {
    expected_exclusive_data[i] = total_exclusive;
    data[i] = (i + 2);
    total += data[i];
    total_exclusive += data[i];
    expected_inclusive_data[i] = total;
  }

  Arcane::Accelerator::impl::SyclScanner<false, Int64, ScannerSumOperator<Int64>> scanner;
  scanner.doScan(queue, data.to1DSmallSpan(), out_data.to1DSmallSpan(), 7);

  const bool do_verbose = (N < 256);
  for (int i = 0; i < N; i++) {
    bool is_bad = out_data[i] != expected_inclusive_data[i];
    if (do_verbose || is_bad)
      std::cout << "OUT_INCL=" << i << " v=" << out_data[i] << " expected=" << expected_inclusive_data[i] << "\n";
    if (is_bad)
      ARCANE_FATAL("Bad value");
  }
  std::cout << "FINAL OUT_INCL=" << (N - 1) << " v=" << out_data[N - 1] << " expected=" << expected_inclusive_data[N - 1] << "\n";

  Arcane::Accelerator::impl::SyclScanner<true, Int64, ScannerSumOperator<Int64>> scanner2;
  scanner2.doScan(queue, data.to1DSmallSpan(), out_data.to1DSmallSpan(), 7);

  for (int i = 0; i < N; i++) {
    bool is_bad = out_data[i] != expected_exclusive_data[i];
    if (do_verbose || is_bad)
      std::cout << "OUT_EXCL=" << i << " v=" << out_data[i] << " expected=" << expected_exclusive_data[i] << "\n";
    if (is_bad)
      ARCANE_FATAL("Bad value");
  }
  std::cout << "FINAL OUT_EXCL=" << (N - 1) << " v=" << out_data[N - 1] << " expected=" << expected_exclusive_data[N - 1] << "\n";
}
