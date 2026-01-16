// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien {
  class MachineInfo
  {
  public :
    MachineInfo(Arccore::MessagePassing::IMessagePassingMng* parellel_mng) ;

    int nbAcceleratorsPerNode() const {
      return m_nb_accelerators_per_node ;
    }

    int nbCoresPerNode() const {
      return m_nb_cores_per_node ;
    }

    int maxNbThreads() const {
      return m_max_nb_threads ;
    }

    int coreId() const {
      return m_core_id ;
    }

    int deviceId() const {
      return m_device_id ;
    }
  private :
    Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr ;
    int m_my_rank = 0 ;
    int m_nb_procs = 1 ;
    int m_nb_cores_per_node = 1 ;
    int m_nb_accelerators_per_node = 0 ;
    int m_max_nb_threads = 1 ;
    int m_core_id = 0 ;
    int m_device_id = -1 ;
  };
}
