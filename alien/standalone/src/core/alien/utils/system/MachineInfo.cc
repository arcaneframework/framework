// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <mpi.h>

#ifdef ALIEN_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <arccore/message_passing/IMessagePassingMng.h>
#include <alien/utils/system/MachineInfo.h>

namespace Alien {

MachineInfo::MachineInfo(Arccore::MessagePassing::IMessagePassingMng* parellel_mng)
: m_parallel_mng(parellel_mng)
{
  m_my_rank = 0 ;
  m_nb_procs = 1 ;
  if(m_parallel_mng)
  {
    m_my_rank = m_parallel_mng->commRank() ;
    m_nb_procs = m_parallel_mng->commSize() ;
  }

#ifdef ALIEN_USE_CUDA
  // Assigner un GPU par processus MPI
  cudaGetDeviceCount(&m_nb_accelerators_per_node);

  if(m_nb_accelerators_per_node>0)
  {
    m_device_id = m_my_rank % m_nb_accelerators_per_node;
  }
#endif
  //cudaSetDevice(device);
}

}
