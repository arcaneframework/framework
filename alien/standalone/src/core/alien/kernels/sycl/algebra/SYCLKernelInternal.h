// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/sycl/SYCLPrecomp.h>

#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
/*---------------------------------------------------------------------------*/

namespace Alien::SYCLInternal
{

#ifndef USE_SYCL2020
  using namespace cl ;
#endif
/*---------------------------------------------------------------------------*/
namespace mi300 {
  // Constantes tuned pour MI300 (gfx942)
  // - CU count  : 228 CUs
  // - Wavefront : 64 threads (RDNA/CDNA)
  // - LDS/CU    : 64 KB  → 1024 threads × 8 B (double) = 8 KB/workgroup, safe
  inline constexpr std::size_t WG_SIZE    = 1024;   // 16 wavefronts/WG
  inline constexpr std::size_t ITEMS_PER_WI = 8;    // unroll : chaque WI traite 8 éléments
}

template <typename T>
class Future
{
 public:
  Future(T& value)
  : m_value(value)
  //, m_d_value{1}
  , m_d_value{ SYCLEnv::instance()->maxNumGroups() }
  {
  }

  T& operator()()
  {
    return m_value;
  }

  T operator()() const
  {
    return m_value;
  }

  T get()
  {
    if (m_parallel_mng) {
      Arccore::MessagePassing::mpWait(m_parallel_mng, m_request);
      m_parallel_mng = nullptr;
    }
    else {
      //auto h_access = m_d_value.template get_access<sycl::access::mode::read>();
      m_event.wait() ;
      auto h_access = m_d_value.get_host_access();
      m_value = h_access[0];
    }
    return m_value;
  }

  sycl::buffer<T, 1>& deviceValue()
  {
    return m_d_value;
  }

  sycl::event& event() {
    return m_event;
  }


  void addRequest(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                  Arccore::MessagePassing::Request request)
  {
    m_parallel_mng = parallel_mng;
    m_request = request;
  }

 private:
  T& m_value;
  sycl::buffer<T, 1> m_d_value;
  sycl::event        m_event;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Arccore::MessagePassing::Request m_request;
};

class KernelInternal
{
 private:
  int m_dot_algo = 3;

 public:
  KernelInternal()
  {
    m_env = SYCLEnv::instance();

    // clang-format off
    m_max_num_groups      = m_env->maxNumGroups() ;
    m_max_work_group_size = m_env->maxWorkGroupSize() ;
    m_total_threads       = m_env->maxNumThreads() ;
    // clang-format on
  }

  virtual ~KernelInternal() {}

  void setDotAlgo(int dot_algo)
  {
    m_dot_algo = dot_algo;
  }

  template <typename T>
  void assign(T const a,
              sycl::buffer<T>& y)
  {
    m_env->internal()->queue().submit(
        [&](sycl::handler& cgh)
        {
          auto acc = y.template get_access<sycl::access::mode::discard_write>(cgh);
          cgh.fill(acc, a);
        });
    /*
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
      m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                         {
                                             auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                             auto y_length = y.size() ;
                                             cgh.parallel_for<class vector_assign>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                                {
                                                                                   auto id = itemId.get_id(0);
                                                                                   for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                      access_y[i] = a;
                                                                                });
                                         });
      // clang-format on
    }*/
  }

  template <typename T, typename Lambda>
  void apply(Lambda const& lambda,
             sycl::buffer<T>& y)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
        m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                           {
                                             auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                             auto y_length = y.size() ;
                                             cgh.parallel_for<class vector_apply>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                                {
                                                                                   auto id = itemId.get_id(0);
                                                                                   for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                      access_y[i] = lambda(i);
                                                                                });
                                           });
      // clang-format on
    }
  }

  template <typename T>
  void scal(T a,
            sycl::buffer<T>& y)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
       m_env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto y_length = y.size() ;
                                           cgh.parallel_for<class vector_scal>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                    access_y[i] = a*access_y[i];
                                                                              });
                                         });
      // clang-format on
    }
  }
#ifdef NAIVE
  template <typename T>
  void axpy(T const a,
            sycl::buffer<T>& x,
            sycl::buffer<T>& y)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
       m_env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto y_length = y.size() ;
                                           cgh.parallel_for<class vector_axpy>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                    access_y[i] += a * access_x[i];
                                                                              });
                                         });
      // clang-format on
    }
  }
#endif
  template <typename T>
  void axpy(T const a,
            sycl::buffer<T>& x,
            sycl::buffer<T>& y)
  {
      using VecT = sycl::vec<T, 2>;

      auto& queue       = m_env->internal()->queue();
      const size_t n    = y.size();
      const size_t n2   = n / 2;           // éléments traités vectoriellement
      const size_t tail = n % 2;

      static constexpr size_t WG_SIZE = 256;
      const size_t num_sm  = queue.get_device()
                                 .get_info<sycl::info::device::max_compute_units>();
      const size_t blocks  = std::max((n2 + WG_SIZE - 1) / WG_SIZE,
                                       num_sm * 4UL);
      const size_t total   = blocks * WG_SIZE;

      // ── Reinterprétation des buffers en vec<T,2> ──────────────────────────
      // Requiert que les buffers soient alignés 16 bytes (garantit par
      // sycl::buffer qui aligne sur max_required_work_group_size).
      sycl::buffer<VecT> x2{ x.template reinterpret<VecT>(sycl::range<1>{n2}) };
      sycl::buffer<VecT> y2{ y.template reinterpret<VecT>(sycl::range<1>{n2}) };

      queue.submit([&](sycl::handler& cgh) {
          auto ax = x2.template get_access<sycl::access::mode::read>(cgh);
          auto ay = y2.template get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for<class vector_axpy_vec>(
              sycl::nd_range<1>{ {total}, {WG_SIZE} },
              [=](sycl::nd_item<1> item) {
                  const size_t stride = item.get_global_range()[0];
                  for (size_t i = item.get_global_id(0); i < n2; i += stride) {
                      // Un seul load 128-bit pour x, un pour y
                      ay[i] = ay[i] + VecT{a, a} * ax[i]; // fused multiply-add
                  }
              });
      });

      // Epilogue scalaire si n est impair (rare en pratique)
      if (tail) {
          sycl::buffer<T> xt{ x.template reinterpret<T>(sycl::range<1>{n}) };
          sycl::buffer<T> yt{ y.template reinterpret<T>(sycl::range<1>{n}) };
          queue.submit([&](sycl::handler& cgh) {
              auto ax = xt.template get_access<sycl::access::mode::read>(cgh);
              auto ay = yt.template get_access<sycl::access::mode::read_write>(cgh);
              cgh.single_task([=]{ ay[n-1] += a * ax[n-1]; });
          });
      }
  }

  template <typename T>
  void axpy(T const a,
            sycl::buffer<T>& x,
            Integer stride_x,
            sycl::buffer<T>& y,
            Integer stride_y)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
       m_env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto x_length = x.size()/stride_x ;
                                           cgh.parallel_for<class vector_axpy>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < x_length; i += itemId.get_range()[0])
                                                                                    access_y[i*stride_y] += a * access_x[i*stride_x];
                                                                              });
                                         });
      // clang-format on
    }
  }
  template <typename T>
  void pointwiseMult(sycl::buffer<T>& x,
                     sycl::buffer<T>& y,
                     sycl::buffer<T>& z)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
       m_env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_y = y.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_z = z.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto y_length = y.size() ;
                                           cgh.parallel_for<class vector_pointwizemult>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                    access_z[i] = access_x[i] * access_y[i];
                                                                              });
                                         });
      // clang-format on
    }
#ifdef PRINT_DEBUG_INFO
    {
      sycl::host_accessor<T, 1, sycl::access::mode::read> x_acc(x);
      sycl::host_accessor<T, 1, sycl::access::mode::read> y_acc(y);
      sycl::host_accessor<T, 1, sycl::access::mode::read> z_acc(z);
      for(int il=0;il<x.size();++il)
      {
        std::cout<<"X Y Z ["<<il<<"] :  "<<x_acc[il]<<"*"<<y_acc[il]<<"="<<z_acc[il]<<std::endl ;
      }
    }
#endif
  }

  template <typename T>
  void copy(sycl::buffer<T>& x,
            sycl::buffer<T>& y)
  {
    //sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
      m_env->internal()->queue().submit(
          [&](sycl::handler& cgh)
         {
           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
           auto access_y = y.template get_access<sycl::access::mode::discard_write>(cgh);
           cgh.copy(access_x,access_y) ;
           /*
           auto y_length = y.size() ;
           cgh.parallel_for<class vector_copy>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                              {
                                                 auto id = itemId.get_id(0);
                                                 for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                    access_y[i] = access_x[i];
                                              });
                                              */
         });
      // clang-format on
    }
  }

  template <typename T>
  void copy(sycl::buffer<T>& x,
            Integer stride_x,
            sycl::buffer<T>& y,
            Integer stride_y)
  {
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
      m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                         {
                                           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto x_length = x.size()/stride_x ;
                                           cgh.parallel_for<class vector_copy>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < x_length; i += itemId.get_range()[0])
                                                                                    access_y[i*stride_y] = access_x[i*stride_x];
                                                                              });
                                         });
      // clang-format on
    }
  }

  template <typename T>
  class sycl_reduction
  {
  };

  // to make global size multiple of local size
  template <typename index_t>
  inline index_t round_up(const index_t x, const index_t y)
  {
    return ((x + y - 1) / y) * y;
  }

  template <typename T>
  class sycl_reduction_sum {};

  template <typename T>
  T reduce_sum(sycl::buffer<T>& x,
               sycl::buffer<T>& y)
  {

    auto& w = getWorkBuffer<T>(x.size());

    // clang-format off
    m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                       {
                                         auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                         auto access_y = y.template get_access<sycl::access::mode::read>(cgh);
                                         //auto access_w = w.template get_access<sycl::access::mode::write>(cgh);
                                         auto access_w = sycl::accessor { w, cgh, sycl::write_only, sycl::property::no_init{}};
                                         //auto access_w = w.get_access(cgh,sycl::write_only, sycl::property::no_init{});
                                         auto y_length = y.size() ;
                                         cgh.parallel_for<class vector_dot>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                            {
                                                                               auto id = itemId.get_id(0);
                                                                               for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                  access_w[i] = access_x[i]*access_y[i];
                                                                            });
                                       });
    // clang-format on

    std::size_t local = m_max_work_group_size;
    std::size_t length = x.size();

    int level = 0;
    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      do {
        auto round_length = round_up(length, local);
        // clang-format off
          auto f = [length, round_length, local, &w](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{round_length},
                                                  sycl::range<1>{local}};
                      auto access_w = w.template get_access<sycl::access::mode::read_write>(h);

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
                      h.parallel_for<class sycl_reduction_sum_T>(range,
                                                            [access_w, scratch, local, length](sycl::nd_item<1> id)
                                                            {
                                                               std::size_t globalid = id.get_global_id(0);
                                                               std::size_t localid = id.get_local_id(0);

                                                              /* All threads collectively read from global memory into local.
                                                               * The barrier ensures all threads' IO is resolved before
                                                               * execution continues (strictly speaking, all threads within
                                                               * a single work-group - there is no co-ordination between
                                                               * work-groups, only work-items). */
                                                               scratch[localid] = (globalid < length)?  access_w[globalid] : 0. ;
                                                               id.barrier(sycl::access::fence_space::local_space);

                                                              /* Apply the reduction operation between the current local
                                                               * id and the one on the other half of the vector. */
                                                              if (globalid < length)
                                                              {
                                                                //int min = (length < local) ? length : local;
                                                                std::size_t min = local ;
                                                                for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                {
                                                                  if (localid < offset)
                                                                  {
                                                                     scratch[localid] += scratch[localid + offset];
                                                                  }
                                                                  id.barrier(sycl::access::fence_space::local_space);
                                                                }
                                                                /* The final result will be stored in local id 0. */
                                                                if (localid == 0)
                                                                {
                                                                  access_w[id.get_group(0)] = scratch[localid];
                                                                }
                                                              }
                                                           });
                 };
        // clang-format on
        m_env->internal()->queue().submit(f);
        /* At this point, you could queue::wait_and_throw() to ensure that
           * errors are caught quickly. However, this would likely impact
           * performance negatively. */
        length = (length + local - 1) / local;
        ++level;
      } while (length > 1);
    }
    //auto h_w = w.template get_access<sycl::access::mode::read>();
    auto h_w = w.get_host_access();
    T sum = h_w[0];
    return sum;
  }

  template <typename T>
  class sycl_map_reduction_sum0 {};

  template <typename T>
  class sycl_map_reduction_sum {};

  template <typename T>
  T map_reduce_sum(sycl::buffer<T>& x,
                   sycl::buffer<T>& y)
  {
    auto& w = getWorkBuffer<T>(x.size());

    std::size_t local = m_max_work_group_size;
    std::size_t length = x.size();

    int level = 0;
    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      //do
      {
        auto round_length = round_up(length, local);
        // clang-format off
          auto f0 = [length, round_length, local, &x,&y, &w](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{round_length},
                                                  sycl::range<1>{local}};
                      auto access_x = x.template get_access<sycl::access::mode::read>(h);
                      auto access_y = y.template get_access<sycl::access::mode::read>(h);
                      //auto access_w = w.template get_access<sycl::access::mode::read_write>(h);
                      auto access_w = sycl::accessor { w, h, sycl::read_write, sycl::property::no_init{}};

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
                      h.parallel_for<class sycl_map_reduction_sum0_T>(range,
                                                                 [access_x,access_y,access_w, scratch, local, length](sycl::nd_item<1> id)
                                                                 {
                                                                   std::size_t globalid = id.get_global_id(0);
                                                                   std::size_t localid = id.get_local_id(0);

                                                                  /* All threads collectively read from global memory into local.
                                                                   * The barrier ensures all threads' IO is resolved before
                                                                   * execution continues (strictly speaking, all threads within
                                                                   * a single work-group - there is no co-ordination between
                                                                   * work-groups, only work-items). */
                                                                   scratch[localid] = (globalid < length)?  access_x[globalid]*access_y[globalid] : 0. ;

                                                                   id.barrier(sycl::access::fence_space::local_space);

                                                                  /* Apply the reduction operation between the current local
                                                                   * id and the one on the other half of the vector. */
                                                                  if (globalid < length)
                                                                  {
                                                                    //int min = (length < local) ? length : local;
                                                                    std::size_t min = local ;
                                                                    for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                    //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                    {
                                                                      if (localid < offset)
                                                                      {
                                                                         scratch[localid] += scratch[localid + offset];
                                                                      }
                                                                      id.barrier(sycl::access::fence_space::local_space);
                                                                    }
                                                                    /* The final result will be stored in local id 0. */
                                                                    if (localid == 0)
                                                                    {
                                                                      access_w[id.get_group(0)] = scratch[localid];
                                                                    }
                                                                  }
                                                                });
                 };
        // clang-format on

        // clang-format off
          auto f1 = [length, round_length, local, &w](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{round_length},
                                                  sycl::range<1>{local}};
                      auto access_w = w.template get_access<sycl::access::mode::read_write>(h);

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
                      h.parallel_for<class sycl_map_reduction_sum_T>(range,
                                                                [access_w, scratch, local, length](sycl::nd_item<1> id)
                                                                {
                                                                   std::size_t globalid = id.get_global_id(0);
                                                                   std::size_t localid = id.get_local_id(0);

                                                                  /* All threads collectively read from global memory into local.
                                                                   * The barrier ensures all threads' IO is resolved before
                                                                   * execution continues (strictly speaking, all threads within
                                                                   * a single work-group - there is no co-ordination between
                                                                   * work-groups, only work-items). */
                                                                   scratch[localid] = (globalid < length)?  access_w[globalid] : 0. ;
                                                                   id.barrier(sycl::access::fence_space::local_space);

                                                                  /* Apply the reduction operation between the current local
                                                                   * id and the one on the other half of the vector. */
                                                                  if (globalid < length)
                                                                  {
                                                                    //int min = (length < local) ? length : local;
                                                                    std::size_t min = local ;
                                                                    for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                    //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                    {
                                                                      if (localid < offset)
                                                                      {
                                                                         scratch[localid] += scratch[localid + offset];
                                                                      }
                                                                      id.barrier(sycl::access::fence_space::local_space);
                                                                    }
                                                                    /* The final result will be stored in local id 0. */
                                                                    if (localid == 0)
                                                                    {
                                                                      access_w[id.get_group(0)] = scratch[localid];
                                                                    }
                                                                  }
                                                                });
                 };
        // clang-format on
        if (level == 0)
          m_env->internal()->queue().submit(f0);
        else
          m_env->internal()->queue().submit(f1);
        /* At this point, you could queue::wait_and_throw() to ensure that
           * errors are caught quickly. However, this would likely impact
           * performance negatively. */
        length = (length + local - 1) / local;
        ++level;
      } //while (length > 1);
    }
    auto h_x = w.get_host_access();
    //T sum = h_x[0] ;
    T sum = 0;
    for (std::size_t i = 0; i < length; ++i)
      sum += h_x[i];
    return sum;
  }

  template <typename T>
  class sycl_map2_reduction_sum0 {};

  template <typename T>
  T map2_reduce_sum(sycl::buffer<T>& x,
                    sycl::buffer<T>& y)
  {
    std::size_t local = m_max_work_group_size;
    std::size_t total_threads = m_total_threads;
    std::size_t length = x.size();

    T sum_init = 0;
    sycl::buffer<T> sum{ &sum_init, 1 };

    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      {
        auto round_length = round_up(length, local);
        // clang-format off
          auto f0 = [length, round_length,total_threads, local, &x, &y, &sum](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{std::min(total_threads,round_length)},
                                                  sycl::range<1>{local}};
                      auto access_x = x.template get_access<sycl::access::mode::read>(h);
                      auto access_y = y.template get_access<sycl::access::mode::read>(h);

                      sycl::accessor access_sum {sum, h};
#ifdef USE_HIPSYCL
                      auto sumReduction = sycl::reduction(access_sum, sycl::plus<T>());
#endif
#ifdef USE_ONEAPI
                      auto sumReduction = sycl::reduction(sum, h, sycl::plus<T>());
#endif
#ifdef USE_ACPPSYCL
                      auto sumReduction = sycl::reduction(sum, h, sycl::plus<T>());
#endif

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
                      h.parallel_for<class sycl_map2_reduction_sum0_T>(range,
                                                                  sumReduction,
                                                                  [access_x,access_y, scratch, local,total_threads,length](sycl::nd_item<1> id, auto &sum)
                                                                  {
                                                                     std::size_t globalid = id.get_global_id(0);
                                                                     std::size_t localid = id.get_local_id(0);

                                                                    /* All threads collectively read from global memory into local.
                                                                     * The barrier ensures all threads' IO is resolved before
                                                                     * execution continues (strictly speaking, all threads within
                                                                     * a single work-group - there is no co-ordination between
                                                                     * work-groups, only work-items). */
                                                                     scratch[localid] = (globalid < length)?  access_x[globalid]*access_y[globalid] : 0. ;
                                                                     for (auto i = globalid+total_threads; i < length; i += total_threads)
                                                                        scratch[localid] += access_x[i]*access_y[i];

                                                                     id.barrier(sycl::access::fence_space::local_space);

                                                                    /* Apply the reduction operation between the current local
                                                                     * id and the one on the other half of the vector. */
                                                                    //if (globalid < length)
                                                                    {
                                                                      //int min = (length < local) ? length : local;
                                                                      std::size_t min = local ;
                                                                      for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                      //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                      {
                                                                        if (localid < offset)
                                                                        {
                                                                           scratch[localid] += scratch[localid + offset];
                                                                        }
                                                                        id.barrier(sycl::access::fence_space::local_space);
                                                                      }
                                                                      /* The final result will be stored in local id 0. */
                                                                      if (localid == 0)
                                                                      {
                                                                        //access_w[id.get_group(0)] = scratch[localid];
                                                                        sum += scratch[localid];
                                                                      }
                                                                    }
                                                                });
                 };
        // clang-format on
        m_env->internal()->queue().submit(f0);
      }
    }
    auto h_sum = sum.get_host_access();
    return h_sum[0];
  }

  template <typename T>
  class sycl_map3_reduction_sum0 {};

  template <typename T>
  T map3_reduce_sum(sycl::buffer<T>& x,
                    sycl::buffer<T>& y)
  {
    std::size_t local = m_max_work_group_size;
    std::size_t total_threads = m_total_threads;
    std::size_t length = x.size();

    //std::vector<T> group_sum(m_max_num_groups) ;
    //sycl::buffer<T> group_sum{ m_max_num_groups };
    auto& group_sum = getWorkBuffer<T>(m_max_num_groups);

    //T sum_init = 0;
    //sycl::buffer<T> sum{ &sum_init, 1 };

    auto round_length = round_up(length, local);
    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      //do
      {
        // clang-format off
          auto f0 = [total_threads,round_length, length, local, &x, &y,&group_sum](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{std::min(total_threads,round_length)},
                                                  sycl::range<1>{local}};
                      auto access_x = x.template get_access<sycl::access::mode::read>(h);
                      auto access_y = y.template get_access<sycl::access::mode::read>(h);

                      auto access_sum = sycl::accessor { group_sum, h, sycl::read_write, sycl::property::no_init{}};

                      //sycl::accessor access_sum {sum, h};
                      //auto sumReduction = sycl::reduction(access_sum, sycl::plus<T>());

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */

                      h.parallel_for<class sycl_map3_reduction_sum0_T>(range,
                                                                  //sumReduction,
                                                                  //[access_x,access_y,scratch,local,length,total_threads] (sycl::nd_item<1> id, auto& sum)
                                                                  [access_x,access_y,access_sum,scratch,local,length,total_threads] (sycl::nd_item<1> id)
                                                                  {
                                                                      std::size_t globalid = id.get_global_id(0);
                                                                      std::size_t localid = id.get_local_id(0);

                                                                      scratch[localid] = (globalid < length)?  access_x[globalid]*access_y[globalid] : 0. ;

                                                                      for (auto i = globalid+total_threads; i < length; i += total_threads)
                                                                        scratch[localid] += access_x[i]*access_y[i];

                                                                      id.barrier(sycl::access::fence_space::local_space);

                                                                      /* Apply the reduction operation between the current local
                                                                       * id and the one on the other half of the vector. */
                                                                      {
                                                                        //int min = (length < local) ? length : local;
                                                                        std::size_t min = local ;
                                                                        for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                        {
                                                                          if (localid < offset)
                                                                          {
                                                                             scratch[localid] += scratch[localid + offset];
                                                                          }
                                                                          id.barrier(sycl::access::fence_space::local_space);
                                                                        }
                                                                      }
                                                                      /* The final result will be stored in local id 0. */
                                                                      if (localid == 0)
                                                                      {
                                                                        access_sum[id.get_group(0)] = scratch[localid];
                                                                        //sum += scratch[localid] ;
                                                                      }
                                                                  });

                 };
        // clang-format on
        m_env->internal()->queue().submit(f0);

      } //while (length > 1);
    }

    //return sum.get_host_access()[0];
    auto h_sum = group_sum.get_host_access();
    T sum = 0;
    for (std::size_t i = 0; i < std::min(total_threads, round_length) / local; ++i)
      sum += h_sum[i];
    return sum;
  }

  template <typename T>
  class map4_reduction_sum {};

  template <typename T>
  void asynch_map4_reduce_sum(sycl::buffer<T>& x,
                             sycl::buffer<T>& y,
                             sycl::buffer<T>& res,
                             sycl::event& event)
  {
    std::size_t local = m_max_work_group_size;
    std::size_t total_threads = m_total_threads;
    std::size_t length = x.size();

    //T sum_init = 0 ;
    //sycl::buffer<T> sum{&sum_init,1};
    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      //do
      {
        auto round_length = round_up(length, local);
        // clang-format off
          auto f0 = [total_threads, round_length, length, local, &x, &y,&res](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{std::min(total_threads,round_length)},
                                                  sycl::range<1>{local}};
                      auto access_x = x.template get_access<sycl::access::mode::read>(h);
                      auto access_y = y.template get_access<sycl::access::mode::read>(h);
#ifdef USE_HIPSYCL
                      sycl::accessor access_sum {res, h};
                      auto sumReduction = sycl::reduction(access_sum, sycl::plus<T>());
#endif
#ifdef USE_ONEAPI
                      auto sumReduction = sycl::reduction(res, h, sycl::plus<T>());
#endif
#ifdef USE_ACPPSYCL
                      auto sumReduction = sycl::reduction(res, h, sycl::plus<T>());
#endif
                      //auto access_sum = sycl::accessor<T,0,access::mode::write,access::target::global_buffer>(res, h);
                      //auto access_sum = sycl::accessor { res, h, sycl::read_write, sycl::property::no_init{}};

                      //auto sumReduction = sycl::reduction(access_sum,sycl::plus<T>(), sycl::property::reduction::initialize_to_identity);
                      //auto sumReduction = sycl::reduction(access_sum,sycl::plus<T>());

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */

                      h.parallel_for<class map4_reduction_sum_T>(range,
                                                            sumReduction,
                                                            [access_x,access_y,scratch,local,length,total_threads] (sycl::nd_item<1> id, auto& sum)
                                                            {
                                                                std::size_t globalid = id.get_global_id(0);
                                                                std::size_t localid = id.get_local_id(0);

                                                                scratch[localid] = (globalid < length)?  access_x[globalid]*access_y[globalid] : 0. ;

                                                                for (auto i = globalid+total_threads; i < length; i += total_threads)
                                                                  scratch[localid] += access_x[i]*access_y[i];

                                                                id.barrier(sycl::access::fence_space::local_space);

                                                                /* Apply the reduction operation between the current local
                                                                 * id and the one on the other half of the vector. */
                                                                //if (globalid < length)
                                                                {
                                                                  //int min = (length < local) ? length : local;
                                                                  std::size_t min = local ;
                                                                  for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                                  //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                  {
                                                                    if (localid < offset)
                                                                    {
                                                                       scratch[localid] += scratch[localid + offset];
                                                                    }
                                                                    id.barrier(sycl::access::fence_space::local_space);
                                                                  }
                                                                }
                                                                /* The final result will be stored in local id 0. */
                                                                if (localid == 0)
                                                                {
                                                                  //access_sum[id.get_group(0)] = scratch[localid];
                                                                  sum += scratch[localid] ;
                                                                }
                                                            });

                 };
        // clang-format on
        event = m_env->internal()->queue().submit(f0);
      }
    }
  }

  template <typename T>
  class map5_reduction_sum {};

  template <typename T>
  class map5_reduction_sum1 {};

  template <typename T>
  void asynch_map5_reduce_sum(sycl::buffer<T>& x,
                             sycl::buffer<T>& y,
                             sycl::buffer<T>& res,
                             sycl::event event)
  {
    std::size_t local = m_max_work_group_size;
    std::size_t total_threads = m_total_threads;
    std::size_t length = x.size();

    int level = 0;
    {
      /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
      do {
        // clang-format off
          auto f0 = [total_threads, length, local, &x, &y,&res](sycl::handler& h) mutable
                   {
                      sycl::nd_range<1> range{sycl::range<1>{total_threads},
                                                  sycl::range<1>{local}};
                      auto access_x = x.template get_access<sycl::access::mode::read>(h);
                      auto access_y = y.template get_access<sycl::access::mode::read>(h);
                      auto access_sum = sycl::accessor { res, h, sycl::read_write, sycl::property::no_init{}};

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */

                      h.parallel_for<class map5_reduction_sum_T>(range,
                                                            [access_x,access_y,access_sum,scratch,local,length,total_threads] (sycl::nd_item<1> id)
                                                            {
                                                                std::size_t globalid = id.get_global_id(0);
                                                                std::size_t localid = id.get_local_id(0);

                                                                scratch[localid] = (globalid < length)?  access_x[globalid]*access_y[globalid] : 0. ;

                                                                for (auto i = globalid+total_threads; i < length; i += total_threads)
                                                                  scratch[localid] += access_x[i]*access_y[i];

                                                                id.barrier(sycl::access::fence_space::local_space);

                                                                /* Apply the reduction operation between the current local
                                                                 * id and the one on the other half of the vector. */
                                                                if (globalid < length)
                                                                {
                                                                  //int min = (length < local) ? length : local;
                                                                  for (std::size_t offset = local / 2; offset > 0; offset /= 2)
                                                                  //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                                  {
                                                                    if (localid < offset)
                                                                    {
                                                                       scratch[localid] += scratch[localid + offset];
                                                                    }
                                                                    id.barrier(sycl::access::fence_space::local_space);
                                                                  }
                                                                }
                                                                /* The final result will be stored in local id 0. */
                                                                if (localid == 0)
                                                                {
                                                                  access_sum[id.get_group(0)] = scratch[localid];
                                                                }

                                                            });

                 };
        // clang-format on
        auto f1 = [length, local, &res](sycl::handler& h) mutable {
          sycl::nd_range<1> range{ sycl::range<1>{ local },
                                       sycl::range<1>{ local } };
          auto access_sum = res.template get_access<sycl::access::mode::read_write>(h);

          //sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>
          //scratch(sycl::range<1>(local), h);
          sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

          /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
          h.parallel_for<class map5_reduction_sum1_T>(range,
                                                 [access_sum, scratch, local, length](sycl::nd_item<1> id) {
                                                   //auto localid = id.get_id(0);
                                                   std::size_t globalid = id.get_global_id(0);
                                                   std::size_t localid = id.get_local_id(0);

                                                   /* All threads collectively read from global memory into local.
                                                                   * The barrier ensures all threads' IO is resolved before
                                                                   * execution continues (strictly speaking, all threads within
                                                                   * a single work-group - there is no co-ordination between
                                                                   * work-groups, only work-items). */
                                                   scratch[localid] = (localid < length) ? access_sum[localid] : 0.;
                                                   id.barrier(sycl::access::fence_space::local_space);

                                                   /* Apply the reduction operation between the current local
                                                                   * id and the one on the other half of the vector. */
                                                   if (localid < length) {
                                                     //int min = (length < local) ? length : local;
                                                     std::size_t min = local;
                                                     for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                     //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                     {
                                                       if (localid < offset) {
                                                         scratch[localid] += scratch[localid + offset];
                                                       }
                                                       id.barrier(sycl::access::fence_space::local_space);
                                                     }
                                                     /* The final result will be stored in local id 0. */
                                                     if (localid == 0) {
                                                       access_sum[0] = scratch[localid];
                                                     }
                                                   }
                                                 });
        };
        // clang-format on
        if (level == 0)
          event = m_env->internal()->queue().submit(f0);
        else
          event = m_env->internal()->queue().submit(f1);

        /* At this point, you could queue::wait_and_throw() to ensure that
           * errors are caught quickly. However, this would likely impact
           * performance negatively. */
        //length = (length + local - 1) / local;
        length = (std::min(total_threads, length) + local - 1) / local;
        ++level;
      } while (length > 1);
    }
    //auto h_x = res.template get_access<sycl::access::mode::read>();
    //T sum = h_x[0] ;
    //T sum = 0;
    //for (int i = 0; i < length; ++i)
    //  sum += h_x[i];
    //return sum;
  }

  template <typename T>
  T reduce_sum2(const std::vector<T>& x)
  {
    T value = 0;
    //for( auto v : x)
    //  value += v ;
    //return value ;

    {
      auto num_groups = m_env->internal()->queue().get_device().get_info<sycl::info::device::max_compute_units>();
      // getting the maximum work group size per thread
      auto work_group_size = m_env->internal()->queue().get_device().get_info<sycl::info::device::max_work_group_size>();
      // building the best number of global thread
      auto total_threads = num_groups * work_group_size;

      std::cout << "LOCAL     =" << work_group_size << std::endl;
      std::cout << "NB GROUPS =" << num_groups << std::endl;
      std::cout << "NB THREADS=" << total_threads << std::endl;

      /* The buffer is used to initialise the data on the device, but we don't
       * want to copy back and trash it. buffer::set_final_data() tells the
       * SYCL runtime where to put the data when the buffer is destroyed; nullptr
       * indicates not to copy back. The vector's length is used as the global
       * work size (unless that is too large). */
      auto device = m_env->internal()->queue().get_device();
      //std::size_t local = std::min(x.size(),device.get_info<sycl::info::device::max_work_group_size>());
      std::size_t local = device.get_info<sycl::info::device::max_work_group_size>();

      std::size_t length = x.size();

      sycl::buffer<T, 1> xbuf(x.data(), sycl::range<1>(x.size()));
      xbuf.set_final_data(nullptr);

      int level = 0;
      {
        /* Each iteration of the do loop applies one level of reduction until
         * the input is of length 1 (i.e. the reduction is complete). */
        do {

          auto round_length = round_up(length, local);
          std::cout << "LENGTH :" << level << " " << length << " " << round_length << " " << local << std::endl;
          // clang-format off
          auto f = [length,round_length,local, &xbuf](sycl::handler& h) mutable
                   {
                      //sycl::nd_range<1> r{sycl::range<1>{std::max(length, local)},
                      //                        sycl::range<1>{std::min(length, local)}};
                      sycl::nd_range<1> r{sycl::range<1>{round_length},
                                              sycl::range<1>{local}};
                      /* Two accessors are used: one to the buffer that is being reduced,
                       * and a second to local memory, used to store intermediate data. */
                      auto x_access = xbuf.template get_access<sycl::access::mode::read_write>(h);

                      //sycl::accessor<T, 1, sycl::access::mode::read_write,sycl::access::target::local>
                      //  scratch(sycl::range<1>(local), h);
                      sycl::local_accessor<T> scratch{sycl::range<1>(local), h};

                      /* The parallel_for invocation chosen is the variant with an nd_item
                       * parameter, since the code requires barriers for correctness. */
                      h.parallel_for<class reduce_sum2>(r, [x_access, scratch, local, length](sycl::nd_item<1> id)
                                                           {
                                                             std::size_t globalid = id.get_global_id(0);
                                                             std::size_t localid = id.get_local_id(0);

                                                            /* All threads collectively read from global memory into local.
                                                             * The barrier ensures all threads' IO is resolved before
                                                             * execution continues (strictly speaking, all threads within
                                                             * a single work-group - there is no co-ordination between
                                                             * work-groups, only work-items). */
                                                             scratch[localid] = (globalid < length)?  x_access[globalid] : 0. ;
                                                             id.barrier(sycl::access::fence_space::local_space);

                                                            /* Apply the reduction operation between the current local
                                                             * id and the one on the other half of the vector. */
                                                            //#ifdef DEBUG
                                                            if (globalid < length)
                                                            {
                                                              //int min = (length < local) ? length : local;
                                                              int min = local ;
                                                              for (std::size_t offset = min / 2; offset > 0; offset /= 2)
                                                              //for (std::size_t offset = id.get_local_range(0) / 2; offset > 0; offset /= 2)
                                                              {
                                                                if (localid < offset)
                                                                {
                                                                   scratch[localid] += scratch[localid + offset];
                                                                }
                                                                id.barrier(sycl::access::fence_space::local_space);
                                                              }
                                                              /* The final result will be stored in local id 0. */
                                                              if (localid == 0)
                                                              {
                                                                x_access[id.get_group(0)] = scratch[localid];
                                                              }
                                                            }
                                                            //#endif
                                                      });
                 };
          // clang-format on
          m_env->internal()->queue().submit(f);
          /* At this point, you could queue::wait_and_throw() to ensure that
           * errors are caught quickly. However, this would likely impact
           * performance negatively. */
          length = (length + local - 1) / local;
          std::cout << "AFTER LENGTH :" << level << " new length" << length << " " << local << std::endl;
          ++level;
        } while (length > 1);
      }
      /* It is always sensible to wrap host accessors in their own scope as
       * kernels using the buffers they access are blocked for the length
       * of the accessor's lifetime. */
      auto hI = xbuf.get_host_acces();
      value = hI[0];
    }
    //value = x[0] ;
    return value;
  }

  template <typename T>
  T sycl_reduce_sum(sycl::buffer<T>& x,
                    sycl::buffer<T>& y)
  {
    T sum_init = 0;
    sycl::buffer<T> sum_buff{ &sum_init, 1 };

    // clang-format off
    m_env->internal()->queue().submit([&](sycl::handler &cgh)
                                      {
                                        auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                        auto access_y = y.template get_access<sycl::access::mode::read>(cgh);
#ifdef USE_HIPSYCL
                                        sycl::accessor sum_acc {sum_buff, cgh};
                                        auto sumReduction = sycl::reduction(sum_acc, sycl::plus<T>());
#endif
#ifdef USE_ONEAPI
                                        auto sumReduction = sycl::reduction(sum_buff, cgh, sycl::plus<T>());
#endif
#ifdef USE_ACPPSYCL
                                        auto sumReduction = sycl::reduction(sum_buff, cgh, sycl::plus<T>());
#endif
                                        cgh.parallel_for(sycl::range<1>{x.size()},
                                                         sumReduction,
                                                         [=](sycl::id<1> idx, auto &sum)
                                                         {
                                                           sum += access_x[idx]*access_y[idx];
                                                         });
                                      });
    // clang-format on

    return sum_buff.get_host_access()[0];
  }

  /*
  namespace mi300 {

  // Constantes tuned pour MI300 (gfx942)
  // - CU count  : 228 CUs
  // - Wavefront : 64 threads (RDNA/CDNA)
  // - LDS/CU    : 64 KB  → 1024 threads × 8 B (double) = 8 KB/workgroup, safe
  inline constexpr std::size_t WG_SIZE    = 1024;   // 16 wavefronts/WG
  inline constexpr std::size_t ITEMS_PER_WI = 8;    // unroll : chaque WI traite 8 éléments

  // ---------------------------------------------------------------------------
  // dot_product_device
  // Calcule <x, y> = sum_i x[i]*y[i] sur GPU MI300.
  // x_buf, y_buf : buffers SYCL en lecture (taille >= n)
  // Retourne le scalaire double via un buffer résultat temporaire.
  // ---------------------------------------------------------------------------
  inline double dot_product(
      sycl::queue& q,
      sycl::buffer<double, 1>& x_buf,
      sycl::buffer<double, 1>& y_buf,
      std::size_t n)
  {
      if (n == 0) return 0.0;

      // Nombre de work-groups : on arrondit au multiple de WG_SIZE*ITEMS_PER_WI
      const std::size_t stride   = WG_SIZE * ITEMS_PER_WI;
      const std::size_t n_wg     = (n + stride - 1) / stride;
      const std::size_t n_padded = n_wg * stride;      // domaine de dispatch

      // Buffer de réductions partielles (une valeur par WG)
      sycl::buffer<double, 1> partial_buf(n_wg);

      // -------------------------------------------------------------------
      // Kernel 1 — réduction locale par work-group
      // Chaque WI charge ITEMS_PER_WI paires (x[i], y[i]) et accumule.
      // La réduction finale dans le WG utilise la LDS (sycl::local_accessor).
      // -------------------------------------------------------------------
      q.submit([&](sycl::handler& cgh) {
          auto x   = x_buf.template get_access<sycl::access::mode::read>(cgh);
          auto y   = y_buf.template get_access<sycl::access::mode::read>(cgh);
          auto out = partial_buf.template get_access<sycl::access::mode::discard_write>(cgh);

          // LDS : WG_SIZE doubles par work-group
          sycl::local_accessor<double, 1> local_sum(WG_SIZE, cgh);

          sycl::nd_range<1> nd_range{n_padded, WG_SIZE};

          cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
              const std::size_t gid    = item.get_global_id(0);
              const std::size_t lid    = item.get_local_id(0);
              const std::size_t wg_id  = item.get_group(0);
              const std::size_t base   = wg_id * stride;

              // Accumulation privée — ITEMS_PER_WI éléments par WI
              double acc = 0.0;
              #pragma unroll
              for (std::size_t k = 0; k < ITEMS_PER_WI; ++k) {
                  const std::size_t idx = base + lid + k * WG_SIZE;
                  if (idx < n) {
                      acc += x[idx] * y[idx];
                  }
              }

              // Dépôt en LDS
              local_sum[lid] = acc;
              sycl::group_barrier(item.get_group());

              // Réduction en arbre dans la LDS (log2(WG_SIZE) = 10 passes)
              for (std::size_t offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                  if (lid < offset) {
                      local_sum[lid] += local_sum[lid + offset];
                  }
                  sycl::group_barrier(item.get_group());
              }

              // WI 0 écrit la somme partielle du WG
              if (lid == 0) {
                  out[wg_id] = local_sum[0];
              }
          });
      });

      // -------------------------------------------------------------------
      // Kernel 2 — réduction finale des n_wg partielles
      // Si n_wg <= WG_SIZE on fait tout en un seul WG.
      // Sinon on rappelle dot_product récursivement (rare pour n < 2^30).
      // -------------------------------------------------------------------
      double result = 0.0;

      if (n_wg <= WG_SIZE) {
          sycl::buffer<double, 1> result_buf(&result, 1);

          q.submit([&](sycl::handler& cgh) {
              auto src = partial_buf.template get_access<sycl::access::mode::read>(cgh);
              auto dst = result_buf.template get_access<sycl::access::mode::discard_write>(cgh);
              sycl::local_accessor<double, 1> lmem(WG_SIZE, cgh);

              // On dispatch exactement WG_SIZE threads, padding avec 0
              cgh.parallel_for(sycl::nd_range<1>{WG_SIZE, WG_SIZE},
                  [=](sycl::nd_item<1> item) {
                      const std::size_t lid = item.get_local_id(0);
                      lmem[lid] = (lid < n_wg) ? src[lid] : 0.0;
                      sycl::group_barrier(item.get_group());

                      for (std::size_t offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                          if (lid < offset)
                              lmem[lid] += lmem[lid + offset];
                          sycl::group_barrier(item.get_group());
                      }
                      if (lid == 0) dst[0] = lmem[0];
                  });
          });
          // L'accès au buffer result_buf en sortie de scope force la synchronisation
      } else {
          // Chemin récursif (n > WG_SIZE² ~ 1M WGs, soit n > ~8G éléments)
          // Rare en pratique — on réduit les partielles via un second appel
          result = dot_product(q, partial_buf, partial_buf, n_wg);
      }

      return result;
  }

  // ---------------------------------------------------------------------------
  // Variante asynchrone : retourne un sycl::event + résultat via buffer externe
  // Utile pour masquer la latence dans une boucle itérative (ex: CG)
  // ---------------------------------------------------------------------------
  inline sycl::event dot_product_async(
      sycl::queue& q,
      sycl::buffer<double, 1>& x_buf,
      sycl::buffer<double, 1>& y_buf,
      sycl::buffer<double, 1>& result_buf,   // taille >= 1
      std::size_t n)
  {
      if (n == 0) {
          return q.submit([&](sycl::handler& cgh) {
              auto r = result_buf.template get_access<sycl::access::mode::discard_write>(cgh);
              cgh.single_task([=]{ r[0] = 0.0; });
          });
      }

      const std::size_t stride   = WG_SIZE * ITEMS_PER_WI;
      const std::size_t n_wg     = (n + stride - 1) / stride;
      const std::size_t n_padded = n_wg * stride;

      // Shared ptr pour que le buffer partiel survive au scope du lambda
      auto partial = std::make_shared<sycl::buffer<double,1>>(n_wg);

      q.submit([&, partial](sycl::handler& cgh) {
          auto x   = x_buf.template get_access<sycl::access::mode::read>(cgh);
          auto y   = y_buf.template get_access<sycl::access::mode::read>(cgh);
          auto out = partial->template get_access<sycl::access::mode::discard_write>(cgh);
          sycl::local_accessor<double, 1> local_sum(WG_SIZE, cgh);

          cgh.parallel_for(sycl::nd_range<1>{n_padded, WG_SIZE},
              [=](sycl::nd_item<1> item) {
                  const std::size_t lid   = item.get_local_id(0);
                  const std::size_t base  = item.get_group(0) * stride;
                  double acc = 0.0;
                  #pragma unroll
                  for (std::size_t k = 0; k < ITEMS_PER_WI; ++k) {
                      const std::size_t idx = base + lid + k * WG_SIZE;
                      if (idx < n) acc += x[idx] * y[idx];
                  }
                  local_sum[lid] = acc;
                  sycl::group_barrier(item.get_group());
                  for (std::size_t offset = WG_SIZE / 2; offset > 0; offset >>= 1) {
                      if (lid < offset) local_sum[lid] += local_sum[lid + offset];
                      sycl::group_barrier(item.get_group());
                  }
                  if (lid == 0) out[item.get_group(0)] = local_sum[0];
              });
      });

      return q.submit([partial](sycl::handler& cgh) {
          auto src = partial->template get_access<sycl::access::mode::read>(cgh);
          // reduction via sycl::reduction (SYCL 2020)
          // Note : result_buf doit être capturé par référence externe
          // → on utilise un single_task pour la somme finale des partielles (n_wg petit)
          cgh.single_task([=](){
              // accessible depuis le lambda — n_wg connu via closure
          });
      });
      // NOTE : pour la variante async complète, préférez sycl::reduction (voir ci-dessous)
  }
  } */

  // ---------------------------------------------------------------------------
  // Variante moderne SYCL 2020 : sycl::reduction (plus concis, potentiellement
  // optimisé par le runtime AdaptiveCPP selon la cible)
  // ---------------------------------------------------------------------------


  inline double dot_product_reduction(sycl::buffer<double, 1>& x_buf,
                                      sycl::buffer<double, 1>& y_buf)
  {
    using namespace mi300 ;
      double result = 0.0;
      std::size_t n = x_buf.size() ;
      auto& q       = m_env->internal()->queue();
      {
          sycl::buffer<double, 1> result_buf(&result, 1);

          const std::size_t stride   = WG_SIZE * ITEMS_PER_WI;
          const std::size_t n_padded = ((n + stride - 1) / stride) * stride;

          q.submit([&](sycl::handler& cgh) {
              auto x   = x_buf.template get_access<sycl::access::mode::read>(cgh);
              auto y   = y_buf.template get_access<sycl::access::mode::read>(cgh);
              auto red = sycl::reduction(result_buf, cgh, sycl::plus<double>{});

              cgh.parallel_for(
                  sycl::nd_range<1>{n_padded, WG_SIZE}, red,
                  [=](sycl::nd_item<1> item, auto& sum) {
                      const std::size_t lid  = item.get_local_id(0);
                      const std::size_t base = item.get_group(0) * stride;
                      double acc = 0.0;
                      #pragma unroll
                      for (std::size_t k = 0; k < ITEMS_PER_WI; ++k) {
                          const std::size_t idx = base + lid + k * WG_SIZE;
                          if (idx < n) acc += x[idx] * y[idx];
                      }
                      sum.combine(acc);
                  });
          });
      } // destruction du result_buf → sync implicite
      return result;
  }

  inline void asynch_dot_product_reduction(sycl::buffer<double, 1>& x_buf,
                                           sycl::buffer<double, 1>& y_buf,
                                           sycl::buffer<double, 1>& result_buf,
                                           sycl::event event)
  {
      using namespace mi300 ;

      std::size_t n = x_buf.size() ;
      auto& q = m_env->internal()->queue();

      const std::size_t stride   = WG_SIZE * ITEMS_PER_WI;
      const std::size_t n_padded = ((n + stride - 1) / stride) * stride;

      event = q.submit([&](sycl::handler& cgh) {
                      auto x   = x_buf.template get_access<sycl::access::mode::read>(cgh);
                      auto y   = y_buf.template get_access<sycl::access::mode::read>(cgh);
                      auto red = sycl::reduction(result_buf, cgh, sycl::plus<double>{});

                      cgh.parallel_for(
                          sycl::nd_range<1>{n_padded, WG_SIZE}, red,
                          [=](sycl::nd_item<1> item, auto& sum) {
                              const std::size_t lid  = item.get_local_id(0);
                              const std::size_t base = item.get_group(0) * stride;
                              double acc = 0.0;
                              #pragma unroll
                              for (std::size_t k = 0; k < ITEMS_PER_WI; ++k) {
                                  const std::size_t idx = base + lid + k * WG_SIZE;
                                  if (idx < n) acc += x[idx] * y[idx];
                              }
                              sum.combine(acc);
                          });
                  });
  }


  template <typename T>
  T dot(sycl::buffer<T>& x,
        sycl::buffer<T>& y)
  {
    switch (m_dot_algo) {
    case 0:
      return reduce_sum(x, y);
    case 1:
      return map_reduce_sum(x, y);
    case 2:
      return map2_reduce_sum(x, y); // with sycl_reduction
    case 3:
      return map3_reduce_sum(x, y);
    case 4: //MI300
      return dot_product_reduction(x, y);
    default:
      return sycl_reduce_sum(x, y);
    }
  }

  template <typename T>
  void dot(sycl::buffer<T>& x,
           sycl::buffer<T>& y,
           sycl::buffer<T>& res,
           sycl::event& event)
{
    switch (m_dot_algo) {
    case 2:
      asynch_map4_reduce_sum(x, y, res, event); // with sycl_reduction
      break;
    case 4:
      asynch_dot_product_reduction(x, y, res, event);
      break;
    default:
      asynch_map5_reduce_sum(x, y, res, event);
      break;
    }
  }

 private:
  // clang-format off
  SYCLEnv*    m_env                 = nullptr ;
  std::size_t m_max_num_groups      = 0 ;
  std::size_t m_max_work_group_size = 0 ;
  std::size_t m_total_threads       = 0 ;
  // clang-format on

  template <typename T>
  sycl::buffer<T>& getWorkBuffer(std::size_t size);

  mutable sycl::buffer<double>* m_double_work = nullptr;
};


/*---------------------------------------------------------------------------*/

} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/
