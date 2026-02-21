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
      auto h_access = m_d_value.get_host_access();
      m_value = h_access[0];
    }
    return m_value;
  }

  sycl::buffer<T, 1>& deviceValue()
  {
    return m_d_value;
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
    }
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
    sycl::range<1> work_items{ m_total_threads };
    {
      // clang-format off
      m_env->internal()->queue().submit( [&](sycl::handler& cgh)
                                         {
                                           auto access_x = x.template get_access<sycl::access::mode::read>(cgh);
                                           auto access_y = y.template get_access<sycl::access::mode::read_write>(cgh);
                                           auto y_length = y.size() ;
                                           cgh.parallel_for<class vector_copy>(sycl::range<1>{m_total_threads}, [=] (sycl::item<1> itemId)
                                                                              {
                                                                                 auto id = itemId.get_id(0);
                                                                                 for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                                                    access_y[i] = access_x[i];
                                                                              });
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
  void map4_reduce_sum(sycl::buffer<T>& x,
                       sycl::buffer<T>& y,
                       sycl::buffer<T>& res)
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
        m_env->internal()->queue().submit(f0);
      }
    }
  }

  template <typename T>
  class map5_reduction_sum {};

  template <typename T>
  class map5_reduction_sum1 {};

  template <typename T>
  void map5_reduce_sum(sycl::buffer<T>& x,
                       sycl::buffer<T>& y,
                       sycl::buffer<T>& res)
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
          m_env->internal()->queue().submit(f0);
        else
          m_env->internal()->queue().submit(f1);

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
    default:
      return sycl_reduce_sum(x, y);
    }
  }

  template <typename T>
  void dot(sycl::buffer<T>& x,
           sycl::buffer<T>& y,
           sycl::buffer<T>& res)
  {
    switch (m_dot_algo) {
    case 2:
      map4_reduce_sum(x, y, res); // with sycl_reduction
      break;
    default:
      map5_reduce_sum(x, y, res);
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
