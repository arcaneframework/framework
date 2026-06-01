// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2LibWrapper.h                                            (C) 2000-2025 */
/*                                                                           */
/* Class that encapsulates the useful functions of the Otf2 library.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_OTF2LIBWRAPPER_H
#define ARCANE_STD_OTF2LIBWRAPPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include "arcane/utils/String.h"

#include "arcane/core/ISubDomain.h"

#include "otf2/otf2.h"

#include <vector>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Wrapper class for OTF2 library functions.
 */
class Otf2LibWrapper
{
 public:

  Otf2LibWrapper(ISubDomain* sub_domain);
  ~Otf2LibWrapper();

  void init(const String& archive_name);
  void finalize();

  OTF2_EvtWriter* getEventWriter();
  uint32_t getEntryPointId(const String& ep_name) const;
  uint32_t getApplicationNameId() const;
  uint32_t getSynchronizeId() const;
  int getMpiRank() const;
  int getMpiNbRank() const;

  static OTF2_TimeStamp getTime();

 private:

  void _createOtf2Ids();
  void _buildOtf2ClockAndStringDefinition(uint64_t global_start_time, uint64_t global_end_time);
  void _buildOtf2ParadigmAndSystemDefinition();
  void _buildOtf2LocationDefinition();
  void _buildOtf2RegionDefinition();
  void _buildOtf2GroupAndCommDefinition();

  static OTF2_FlushType _preFlush(void* user_data, OTF2_FileType file_type, OTF2_LocationRef location, void* caller_data, bool final);
  static OTF2_TimeStamp _postFlush(void* user_data, OTF2_FileType file_type, OTF2_LocationRef location);

 public:

  // A simple structure to associate an entry point name with an identifier
  struct EntryPointId
  {
    EntryPointId(String name, Integer id)
    : m_name(name)
    , m_id(id)
    {}
    String m_name;
    Integer m_id;
    // Functor to allow "heterogeneous lookup" from name to ID
    struct EntryPointIdCompare
    {
      using is_transparent = void;
      bool operator()(const EntryPointId& lhs, const EntryPointId& rhs) const
      {
        return lhs.m_name < rhs.m_name;
      }
      bool operator()(const String& name, const EntryPointId& ep_id) const
      {
        return name < ep_id.m_name;
      }
      bool operator()(const EntryPointId& ep_id, const String& name) const
      {
        return ep_id.m_name < name;
      }
    };
  };

 private:

  // Internal structure for managing indices and offsets...
  struct InternalIds
  {
    uint32_t m_desc_offset; // Offset for descriptions of MPI operation names
    uint32_t m_empty_id; // ID for the empty string
    uint32_t m_thread_id; // ID for the CPU thread name mapped to the MPI process
    uint32_t m_hostname_id; // ID for the machine name
    uint32_t m_class_id; // ID for the machine class
    uint32_t m_comm_world_id; // ID for the MPI_COMM_WOLRD
    uint32_t m_mpi_id; // ID for the MPI string
    uint32_t m_comm_id; // ID for the MPI paradigm string
    uint32_t m_win_id; // ID for the MPI paradigm string
    uint32_t m_comm_self_id; // ID for the "comme self" string
    uint32_t m_ep_id; // ID for the Entry Point string
    uint32_t m_sync_id; // ID for the Synchronize string
    uint32_t m_app_name; // ID for the application name
    uint32_t m_rank_offset; // Offset for MPI ranks
    std::set<EntryPointId, EntryPointId::EntryPointIdCompare> m_ep_id_set;
    // Set of entry point names and their IDs
  };

  ISubDomain* m_sub_domain = nullptr;
  OTF2_Archive* m_archive = nullptr;
  OTF2_FlushCallbacks m_flush_callbacks;
  OTF2_EvtWriter* m_evt_writer = nullptr;
  uint64_t m_evt_nb = 0;
  OTF2_GlobalDefWriter* m_global_def_writer = nullptr;
  static uint64_t s_epoch_start;
  InternalIds m_id;
  std::vector<uint64_t> m_comm_members;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
