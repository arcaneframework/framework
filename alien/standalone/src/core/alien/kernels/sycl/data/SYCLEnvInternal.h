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


namespace Alien
{

namespace SYCLInternal
{
#ifndef USE_SYCL2020
  using namespace cl ;
#endif

  struct ALIEN_EXPORT EnvInternal
  {
    EnvInternal()
    : m_queue(sycl::gpu_selector{})
    {
      printPlatformInfo();

      auto device = m_queue.get_device();

      m_max_num_groups = m_queue.get_device().get_info<sycl::info::device::max_compute_units>();
      // getting the maximum work group size per thread
      m_max_work_group_size = m_queue.get_device().get_info<sycl::info::device::max_work_group_size>();
      m_subgroup_size = m_queue.get_device().get_info<sycl::info::device::sub_group_sizes>()[0];
      m_max_num_subgroups_per_group = m_max_work_group_size/m_subgroup_size ;
      m_max_num_threads = m_max_num_groups * m_max_work_group_size;

      std::cout << "========== SYCL QUEUE INFO ===============" << std::endl;
      std::cout<< " DEVICE NAME         = " << m_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
      std::cout << "MAX NB GROUPS       = " << m_max_num_groups << std::endl;
      std::cout << "MAX WORK GROUP SIZE = " << m_max_work_group_size << std::endl;
      std::cout << "SUB GROUP SIZE      = " << m_subgroup_size << std::endl ;
      std::cout << "MAX NB SUBGROUPs PER GROUP = " << m_max_num_subgroups_per_group << std::endl;
      std::cout << "MAX NB THREADS      = " << m_max_num_threads << std::endl;
    }

    sycl::queue& queue()
    {
      return m_queue;
    }

    std::size_t maxNumGroups()
    {
      return m_max_num_groups;
    }

    std::size_t maxWorkGroupSize()
    {
      return m_max_work_group_size;
    }

    std::size_t maxNumThreads()
    {
      return m_max_num_threads;
    }

    int printPlatformInfo()
    {
      // Loop over all available SYCL platforms.
      for (const sycl::platform& platform :
           sycl::platform::get_platforms()) {

        // Print some information about the platform.
        std::cout << "============ Platform ============" << std::endl;
        std::cout << " Name   : "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;
        std::cout << " Vendor : "
                  << platform.get_info<sycl::info::platform::vendor>()
                  << std::endl;
        std::cout << " Version: "
                  << platform.get_info<sycl::info::platform::version>()
                  << std::endl;

        // Loop over all devices available from this platform.
        for (const sycl::device& device : platform.get_devices()) {

          // Print some information about the device.
          std::cout << "------------- Device -------------" << std::endl;
          std::cout << " Name   : "
                    << device.get_info<sycl::info::device::name>()
                    << std::endl;
          std::cout << " Vendor : "
                    << device.get_info<sycl::info::device::vendor>()
                    << std::endl;
          std::cout << " Version: "
                    << device.get_info<sycl::info::device::version>()
                    << std::endl;
        }
      }
      return 0;
    }

    // clang-format off
    //sycl::default_selector     m_device_selector;
    sycl::queue                m_queue;
    std::size_t                m_max_num_groups              = 0 ;
    std::size_t                m_max_work_group_size         = 0 ;
    std::size_t                m_subgroup_size               = 0 ;
    std::size_t                m_max_num_subgroups_per_group = 0 ;
    std::size_t                m_max_num_threads             = 0 ;
    // clang-format on
  };
} // namespace SYCLInternal
} // namespace Alien
