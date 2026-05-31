// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2LibWrapper.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Class that encapsulates the useful functions of the Otf2 library.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/Otf2LibWrapper.h"

#include "arcane/utils/Collection.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IDirectory.h"
#include "arcane/IParallelMng.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/IApplication.h"

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/internal/MessagePassingMpiEnum.h"
#include "arccore/base/PlatformUtils.h"

#include <otf2/OTF2_MPI_Collectives.h>
#include <numeric>
#include <filesystem>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore::MessagePassing::Mpi;

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Static member
uint64_t Otf2LibWrapper::s_epoch_start = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constructor.
Otf2LibWrapper::
Otf2LibWrapper(ISubDomain* sub_domain)
: m_sub_domain(sub_domain)
{
	m_flush_callbacks.otf2_pre_flush = _preFlush;
	m_flush_callbacks.otf2_post_flush = _postFlush;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Destructor.
Otf2LibWrapper::
~Otf2LibWrapper()
{
	// Close the archive
	if (m_archive)
	  OTF2_Archive_Close(m_archive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Initialization method. Allows defining the path where the archive will be found, as well as its name
void Otf2LibWrapper::
init(const String& archive_name)
{
	// Alias on the MPI comm
	MPI_Comm* mpi_comm((MPI_Comm*)m_sub_domain->parallelMng()->getMPICommunicator());
	if (!mpi_comm)
		ARCANE_FATAL("Impossible d'initialiser la librairie Otf2 sans communicateur MPI");

	// Initialization of the array of participants in the comm (i.e., the sub-domain IDs, etc.)
	m_comm_members.resize(m_sub_domain->nbSubDomain());
  std::iota(m_comm_members.begin(), m_comm_members.end(), 0);

	// Check for the existence of files and delete them if necessary
	String dir_name(m_sub_domain->listingDirectory().path() + "/" + archive_name);
	String otf2_name(dir_name + ".otf2");
	String def_name(dir_name + ".def");
	if (m_sub_domain->parallelMng()->isMasterIO() && std::filesystem::exists(dir_name.localstr())) {
		std::filesystem::remove_all(dir_name.localstr());
		std::filesystem::remove_all(otf2_name.localstr());
		std::filesystem::remove_all(def_name.localstr());
	}
	// Sync before opening
	MPI_Barrier(*mpi_comm);

	// Save the start time to synchronize the recordings of each MPI rank
	s_epoch_start = getTime();

	// Open the archive
	m_archive = OTF2_Archive_Open(m_sub_domain->listingDirectory().path().localstr(),
			                          archive_name.localstr(), OTF2_FILEMODE_WRITE,
	                              1024 * 1024 /* event chunk size */,
	                              4 * 1024 * 1024 /* def chunk size */,
	                              OTF2_SUBSTRATE_POSIX, OTF2_COMPRESSION_NONE);
	if (!m_archive)
		ARCANE_FATAL("Impossible de creer l'archive OTF2");

	// Attaching the callbacks
	OTF2_Archive_SetFlushCallbacks(m_archive, &m_flush_callbacks, NULL);
	if (OTF2_MPI_Archive_SetCollectiveCallbacks(m_archive, *mpi_comm, MPI_COMM_NULL) != OTF2_SUCCESS)
		ARCANE_FATAL("Probleme lors du positionnement des callbacks MPI pour la librairie OTF2");

	// Initialize event files (not yet created)
	OTF2_Archive_OpenEvtFiles(m_archive);

	// Initialize the event writer
	m_evt_writer = OTF2_Archive_GetEvtWriter(m_archive, m_sub_domain->subDomainId());

	// Creation of IDs for the archive definitions
	_createOtf2Ids();

	// DBG test of the set content
	/*
	m_sub_domain->traceMng()->info() << "===== EntryPointIdSet =====";
	for (auto i : m_id.m_ep_id_set)
		m_sub_domain->traceMng()->info() << "{" << i.m_name << ", " << i.m_id << "}";
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Method to call to finalize the archive creation (i.e., we no longer want to record events)
void Otf2LibWrapper::
finalize()
{
	// Alias on the comm
	MPI_Comm* mpi_comm((MPI_Comm*)m_sub_domain->parallelMng()->getMPICommunicator());

	// Save the end time to synchronize the recordings of each MPI rank
	uint64_t epoch_end(getTime());

	// Close what is related to events
	OTF2_Archive_CloseEvtWriter(m_archive, m_evt_writer);
  OTF2_Archive_CloseEvtFiles(m_archive);

	// Temporary storage to retrieve global timings
	uint64_t sync_epoch;

	// Exchanges to get the global time window
	// Smallest time for the start
	/*
	MPI_Reduce(&m_epoch_start, &sync_epoch, 1, OTF2_MPI_UINT64_T, MPI_MIN,
			       m_sub_domain->parallelMng()->masterIORank(), *mpi_comm);
	std::swap(m_epoch_start, sync_epoch);
	*/
	// Largest time for the end
	MPI_Reduce(&epoch_end, &sync_epoch, 1, OTF2_MPI_UINT64_T, MPI_MAX,
			       m_sub_domain->parallelMng()->masterIORank(), *mpi_comm);
	std::swap(epoch_end, sync_epoch);

	// We finish creating the definitions for the archive
	//_buildOtf2ClockAndStringDefinition(m_epoch_start, epoch_end);
	_buildOtf2ClockAndStringDefinition(0, epoch_end);
	_buildOtf2ParadigmAndSystemDefinition();
	_buildOtf2LocationDefinition();
	_buildOtf2RegionDefinition();
	_buildOtf2GroupAndCommDefinition();  // The effective writing happens in this method

	// Sync before closing
	MPI_Barrier(*mpi_comm);

	// Close the archive file
	OTF2_Archive_Close(m_archive);
	m_archive = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Accessor for the internal otf2 event writer.
//! Increments an internal event counter on each call.
OTF2_EvtWriter* Otf2LibWrapper::
getEventWriter()
{
	++m_evt_nb;
	return m_evt_writer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Helper for the ID of an entry point via its name.
uint32_t Otf2LibWrapper::
getEntryPointId(const String& ep_name) const
{
	auto id(m_id.m_ep_id_set.find(ep_name));
	if (id == m_id.m_ep_id_set.end())
		ARCANE_FATAL(String("Impossible de trouver le point d'entree ") + ep_name);
	return id->m_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Helper for the MPI rank number
int Otf2LibWrapper::
getMpiRank() const
{
	return m_sub_domain->subDomainId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Helper for the number of MPI ranks
int Otf2LibWrapper::
getMpiNbRank() const
{
	return m_sub_domain->nbSubDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Helper for the application name
uint32_t Otf2LibWrapper::
getApplicationNameId() const
{
	return m_id.m_app_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Helper for the string "synchronize"
uint32_t Otf2LibWrapper::
getSynchronizeId() const
{
	return m_id.m_sync_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal static method to retrieve the timestamp.
OTF2_TimeStamp Otf2LibWrapper::
getTime()
{
	// We use MPI_Wtime to get timestamps for our events but need to convert the seconds to an integral value.
	// We use a nano second resolution.
	double t(MPI_Wtime() * 1e9);
	return (uint64_t)t - Otf2LibWrapper::s_epoch_start;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal static method to set the callback to be called before the event is recorded
OTF2_FlushType Otf2LibWrapper::
_preFlush([[maybe_unused]] void* user_data, [[maybe_unused]] OTF2_FileType file_type,
		      [[maybe_unused]] OTF2_LocationRef location, [[maybe_unused]] void* caller_data,
		      [[maybe_unused]] bool final)
{
	// The pre flush callback is triggered right before a buffer flush.
	// It needs to return either OTF2_FLUSH to flush the recorded data to a file
	// or OTF2_NO_FLUSH to suppress flushing data to a file
	return OTF2_FLUSH;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal static method to set the callback to be called after the event is recorded
OTF2_TimeStamp Otf2LibWrapper::
_postFlush([[maybe_unused]] void* user_data, [[maybe_unused]] OTF2_FileType file_type,
           [[maybe_unused]] OTF2_LocationRef location)
{
	// The post flush callback is triggered right after a memory buffer flush.
	// It has to return a current timestamp which is recorded to mark the time spend in a buffer flush
	return getTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Internal method for creating naming identifiers for
//! the definitions necessary for the OTF2 archive.
//! Must be called in init
void Otf2LibWrapper::
_createOtf2Ids()
{
	// The IDs of the MPI operation names are linked to the enum value
	// Their description is offset accordingly
	// We use a counter to increment the identifiers
	u_int32_t offset(static_cast<uint32_t>(eMpiName::NameOffset));
	m_id.m_desc_offset = offset;

	// We continue with the rest; I haven't found a brilliant idea for a smart index...
	offset *= 2;
	// Empty string: 2 x eMpiName::NameOffset
	m_id.m_empty_id = offset++;
	// CPU thread name mapped to the MPI process: 2 x eMpiName::NameOffset + 1
	m_id.m_thread_id = offset++;
	// Machine name: 2 x eMpiName::NameOffset + 2
	m_id.m_hostname_id = offset++;
	// Machine class... not inspired...: 2 x eMpiName::NameOffset + 3
	m_id.m_class_id = offset++;
	// MPI Communicator: 2 x eMpiName::NameOffset + 4
	m_id.m_comm_world_id = offset++;
	// MPI: 2 x eMpiName::NameOffset + 5
	m_id.m_mpi_id = offset++;
	// Comm ID: 2 x eMpiName::NameOffset + 6
	m_id.m_comm_id = offset++;
	// Win ID: 2 x eMpiName::NameOffset + 7
	m_id.m_win_id = offset++;
	// MPI_COMM_SELF: 2 x eMpiName::NameOffset + 8
	m_id.m_comm_self_id = offset++;
	// Entry Point: 2 x eMpiName::NameOffset + 9
	m_id.m_ep_id = offset++;
  // Synchronize: 2 x eMpiName::NameOffset + 10
  m_id.m_sync_id = offset++;
	// Application name: 2 x eMpiName::NameOffset + 11
	m_id.m_app_name = offset++;
	// We continue with the MPI rank names: starting from 2 x eMpiName::NameOffset + 12
	m_id.m_rank_offset = offset;
	offset += m_sub_domain->nbSubDomain();

  // We finish with the entry points: starting from 2 x eMpiName::NameOffset + 12 + nbSubDomain()
  for (auto i : m_sub_domain->timeLoopMng()->usedTimeLoopEntryPoints()) {
		// We add the entry point reference to the table
    // NOTE: The ID used must be identical to the one in Otf2MessagePassingProfilingService
		m_id.m_ep_id_set.emplace(EntryPointId(i->fullName(), offset));
		// We increment for everyone
		offset++;
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal method for creating the set of strings used in
//! the definitions necessary for the OTF2 archive.
//! Must be called at the beginning of archiving
void Otf2LibWrapper::
_buildOtf2ClockAndStringDefinition(uint64_t global_start_time, uint64_t global_end_time)
{
	/*
	 * Only the sub-domain responsible for I/Os will fill this OTF2 structure
	 * (because we don't know much about what's underneath, but probably too many chars*)
	 * and to be consistent with the identifiers associated with the entry point names, we communicate the start of the entry point IDs.
	 */

	// Only the responsible sub-domain writes the otf2 archive
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	// Retrieval of the writer
	m_global_def_writer = OTF2_Archive_GetGlobalDefWriter(m_archive);

	// Definition of temporal properties
  // A new argument is available starting from version 3.0. It is the last argument
  // *  @param realtimeTimestamp A realtime timestamp of the `globalOffset` timestamp
  // *                           in nanoseconds since 1970-01-01T00:00 UTC. Use
  // *                           @eref{OTF2_UNDEFINED_TIMESTAMP} if no such
  // *                           timestamp exists. Since version 3.0.
  OTF2_GlobalDefWriter_WriteClockProperties(m_global_def_writer, 1000000000,
                                            global_start_time, global_end_time - global_start_time + 1
#if OTF2_VERSION_MAJOR >= 3
                                            ,OTF2_UNDEFINED_TIMESTAMP
#endif
                                            );

	// Definition of all names that will appear in tools capable of reading the otf2 archive
	// We start with the names of the MPI operations
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Bcast), MpiInfo(eMpiName::Bcast).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Gather), MpiInfo(eMpiName::Gather).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Gatherv), MpiInfo(eMpiName::Gatherv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allgather), MpiInfo(eMpiName::Allgather).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allgatherv), MpiInfo(eMpiName::Allgatherv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Scatterv), MpiInfo(eMpiName::Scatterv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Alltoall), MpiInfo(eMpiName::Alltoall).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Alltoallv), MpiInfo(eMpiName::Alltoallv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Barrier), MpiInfo(eMpiName::Barrier).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Reduce), MpiInfo(eMpiName::Reduce).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allreduce), MpiInfo(eMpiName::Allreduce).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Scan), MpiInfo(eMpiName::Scan).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Sendrecv), MpiInfo(eMpiName::Sendrecv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Isend), MpiInfo(eMpiName::Isend).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Send), MpiInfo(eMpiName::Send).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Irecv), MpiInfo(eMpiName::Irecv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Recv), MpiInfo(eMpiName::Recv).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Test), MpiInfo(eMpiName::Test).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Probe), MpiInfo(eMpiName::Probe).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Get_count), MpiInfo(eMpiName::Get_count).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Wait), MpiInfo(eMpiName::Wait).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Waitall), MpiInfo(eMpiName::Waitall).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Testsome), MpiInfo(eMpiName::Testsome).name().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Waitsome), MpiInfo(eMpiName::Waitsome).name().localstr());
  // We use a counter to increment the identifiers
  u_int32_t offset(static_cast<uint32_t>(eMpiName::NameOffset));

  // We continue with the descriptions of these operations. We use the offset value of the enum field count.
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Bcast) + offset, MpiInfo(eMpiName::Bcast).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Gather) + offset, MpiInfo(eMpiName::Gather).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Gatherv) + offset, MpiInfo(eMpiName::Gatherv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allgather) + offset, MpiInfo(eMpiName::Allgather).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allgatherv) + offset, MpiInfo(eMpiName::Allgatherv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Scatterv) + offset, MpiInfo(eMpiName::Scatterv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Alltoall) + offset, MpiInfo(eMpiName::Alltoall).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Alltoallv) + offset, MpiInfo(eMpiName::Alltoallv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Barrier) + offset, MpiInfo(eMpiName::Barrier).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Reduce) + offset, MpiInfo(eMpiName::Reduce).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Allreduce) + offset, MpiInfo(eMpiName::Allreduce).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Scan) + offset, MpiInfo(eMpiName::Scan).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Sendrecv) + offset, MpiInfo(eMpiName::Sendrecv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Isend) + offset, MpiInfo(eMpiName::Isend).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Send) + offset, MpiInfo(eMpiName::Send).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Irecv) + offset, MpiInfo(eMpiName::Irecv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Recv) + offset, MpiInfo(eMpiName::Recv).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Test) + offset, MpiInfo(eMpiName::Test).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Probe) + offset, MpiInfo(eMpiName::Probe).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Get_count) + offset, MpiInfo(eMpiName::Get_count).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Wait) + offset, MpiInfo(eMpiName::Wait).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Waitall) + offset, MpiInfo(eMpiName::Waitall).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Testsome) + offset, MpiInfo(eMpiName::Testsome).description().localstr());
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer,
	                                 static_cast<uint32_t>(eMpiName::Waitsome) + offset, MpiInfo(eMpiName::Waitsome).description().localstr());

	// On continu avec le reste, j'ai pas trouve d'idee geniale pour avoir un indice intelligent ...
	offset *= 2;
	// Chaine vide : 2 x eMpiName::NameOffset
  OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "");
	// Nom du thread CPU mappe sur le process MPI : 2 x eMpiName::NameOffset + 1
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Master Thread");
	// Nom de la machine : 2 x eMpiName::NameOffset + 2
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, Arccore::Platform::getHostName().localstr());
	// Classe de la machine... pas inspire... : 2 x eMpiName::NameOffset + 3
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Compute node");
	// MPI Communicator : 2 x eMpiName::NameOffset + 4
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "MPI_COMM_WORLD");
	// MPI : 2 x eMpiName::NameOffset + 5
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "MPI");

	// Comm id: 2 x eMpiName::NameOffset + 6
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Comm ${id}");
	// + 7
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Win ${id}");
	// + 8
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "MPI_COMM_SELF");

	// Entry Point : 2 x eMpiName::NameOffset + 9
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Entry Point");
	// Synchronize : 2 x eMpiName::NameOffset + 10
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, "Synchronize");
	// Application name : 2 x eMpiName::NameOffset + 11
	OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, m_sub_domain->application()->applicationName().localstr());
	// On continu avec les noms des rangs MPI
	for (Int32 i(0); i < m_sub_domain->nbSubDomain(); ++i) {
		String mpi_rank_name(std::string("MPI Rank ") + std::to_string(i));
		OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, mpi_rank_name.localstr());
	}
	// On termine avec les points d'entree
	for (auto i : m_sub_domain->timeLoopMng()->usedTimeLoopEntryPoints())
		// On cree la definition otf2 pour le responsable des IOs
		OTF2_GlobalDefWriter_WriteString(m_global_def_writer, offset++, i->fullName().localstr());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal method to define the regions associated with otf2 events
//! Must be called at the end of archiving
void Otf2LibWrapper::
_buildOtf2ParadigmAndSystemDefinition()
{
	// Seul le sous-domaine responsable ecrit l'archive otf2
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	OTF2_GlobalDefWriter_WriteParadigm(m_global_def_writer, OTF2_PARADIGM_MPI, m_id.m_mpi_id, OTF2_PARADIGM_CLASS_PROCESS);
	OTF2_AttributeValue attr_val;
	attr_val.stringRef = m_id.m_comm_id;
	OTF2_GlobalDefWriter_WriteParadigmProperty(m_global_def_writer, OTF2_PARADIGM_MPI,
	                                           OTF2_PARADIGM_PROPERTY_COMM_NAME_TEMPLATE, OTF2_TYPE_STRING, attr_val);

	attr_val.stringRef = m_id.m_win_id;
	OTF2_GlobalDefWriter_WriteParadigmProperty(m_global_def_writer, OTF2_PARADIGM_MPI,
	                                           OTF2_PARADIGM_PROPERTY_RMA_WIN_NAME_TEMPLATE, OTF2_TYPE_STRING, attr_val);

	// Definition du systeme
	OTF2_GlobalDefWriter_WriteSystemTreeNode(m_global_def_writer, 0 /* id */, m_id.m_hostname_id, m_id.m_class_id,
	                                         OTF2_UNDEFINED_SYSTEM_TREE_NODE /* parent */);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal method to write the system definition associated with the otf2 archive
void Otf2LibWrapper::
_buildOtf2LocationDefinition()
{
  // On a besoin de connaitre le nb d'evenement par sous-domaine
  UniqueArray<Integer> recv_buffer(m_sub_domain->nbSubDomain());
  UniqueArray<Integer> send_buffer(1, static_cast<Integer >(m_evt_nb));
  m_sub_domain->parallelMng()->gather(send_buffer, recv_buffer, m_sub_domain->parallelMng()->masterIORank());

	// Seul le sous-domaine responsable ecrit l'archive otf2
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	// Location group pour tous les ranks MPI
	for (Int32 i(0); i < m_sub_domain->nbSubDomain(); ++i) {
    OTF2_GlobalDefWriter_WriteLocationGroup(m_global_def_writer, i /* id */,
                                            m_id.m_rank_offset + i /* name */,
                                            OTF2_LOCATION_GROUP_TYPE_PROCESS,
                                            0 /* system tree */
#if OTF2_VERSION_MAJOR >= 3
                                            ,OTF2_UNDEFINED_LOCATION_GROUP
#endif
                                            );
	}
	for (Int32 i(0); i < m_sub_domain->nbSubDomain(); ++i) {
		OTF2_GlobalDefWriter_WriteLocation(m_global_def_writer, i /* id */, m_id.m_thread_id /* name */,
		                                   OTF2_LOCATION_TYPE_CPU_THREAD, recv_buffer.at(i) /* # events */,
		                                   i /* location group */);
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal method to define the regions associated with otf2 events
//! Must be called at the end of archiving
void Otf2LibWrapper::
_buildOtf2RegionDefinition()
{
	// Seul le sous-domaine responsable ecrit l'archive otf2
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	// Definition des regions
	// broadcast
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Bcast),
	                                 static_cast<uint32_t>(eMpiName::Bcast), static_cast<uint32_t>(eMpiName::Bcast),
	                                 static_cast<uint32_t>(eMpiName::Bcast) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ONE2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// gather
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Gather),
	                                 static_cast<uint32_t>(eMpiName::Gather), static_cast<uint32_t>(eMpiName::Gather),
	                                 static_cast<uint32_t>(eMpiName::Gather) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ONE,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// gather variable
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Gatherv),
	                                 static_cast<uint32_t>(eMpiName::Gatherv), static_cast<uint32_t>(eMpiName::Gatherv),
	                                 static_cast<uint32_t>(eMpiName::Gatherv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ONE,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// all gather
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Allgather),
	                                 static_cast<uint32_t>(eMpiName::Allgather), static_cast<uint32_t>(eMpiName::Allgather),
	                                 static_cast<uint32_t>(eMpiName::Allgather) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// all gather variable
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Allgatherv),
	                                 static_cast<uint32_t>(eMpiName::Allgatherv), static_cast<uint32_t>(eMpiName::Allgatherv),
	                                 static_cast<uint32_t>(eMpiName::Allgatherv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// scatter variable
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Scatterv),
	                                 static_cast<uint32_t>(eMpiName::Scatterv), static_cast<uint32_t>(eMpiName::Scatterv),
	                                 static_cast<uint32_t>(eMpiName::Scatterv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ONE2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// all to all
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Alltoall),
	                                 static_cast<uint32_t>(eMpiName::Alltoall), static_cast<uint32_t>(eMpiName::Alltoall),
	                                 static_cast<uint32_t>(eMpiName::Alltoall) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// all to all variable
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Alltoallv),
	                                 static_cast<uint32_t>(eMpiName::Alltoallv), static_cast<uint32_t>(eMpiName::Alltoallv),
	                                 static_cast<uint32_t>(eMpiName::Alltoallv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// barrier
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Barrier),
	                                 static_cast<uint32_t>(eMpiName::Barrier), static_cast<uint32_t>(eMpiName::Barrier),
	                                 static_cast<uint32_t>(eMpiName::Barrier) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_BARRIER,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// reduce
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Reduce),
	                                 static_cast<uint32_t>(eMpiName::Reduce), static_cast<uint32_t>(eMpiName::Reduce),
	                                 static_cast<uint32_t>(eMpiName::Reduce) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ONE,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// all reduce
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Allreduce),
	                                 static_cast<uint32_t>(eMpiName::Allreduce), static_cast<uint32_t>(eMpiName::Allreduce),
	                                 static_cast<uint32_t>(eMpiName::Allreduce) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_ALL2ALL,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// scan
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Scan),
	                                 static_cast<uint32_t>(eMpiName::Scan), static_cast<uint32_t>(eMpiName::Scan),
	                                 static_cast<uint32_t>(eMpiName::Scan) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_COLL_OTHER,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// sendrecv
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Sendrecv),
	                                 static_cast<uint32_t>(eMpiName::Sendrecv), static_cast<uint32_t>(eMpiName::Sendrecv),
	                                 static_cast<uint32_t>(eMpiName::Sendrecv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// non blocking send
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Isend),
	                                 static_cast<uint32_t>(eMpiName::Isend), static_cast<uint32_t>(eMpiName::Isend),
	                                 static_cast<uint32_t>(eMpiName::Isend) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// blocking send
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Send),
	                                 static_cast<uint32_t>(eMpiName::Send), static_cast<uint32_t>(eMpiName::Send),
	                                 static_cast<uint32_t>(eMpiName::Send) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// non blocking recv
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Irecv),
	                                 static_cast<uint32_t>(eMpiName::Irecv), static_cast<uint32_t>(eMpiName::Irecv),
	                                 static_cast<uint32_t>(eMpiName::Irecv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// blocking recv
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Recv),
	                                 static_cast<uint32_t>(eMpiName::Recv), static_cast<uint32_t>(eMpiName::Recv),
	                                 static_cast<uint32_t>(eMpiName::Recv) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// test
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Test),
	                                 static_cast<uint32_t>(eMpiName::Test), static_cast<uint32_t>(eMpiName::Test),
	                                 static_cast<uint32_t>(eMpiName::Test) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// probe
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Probe),
	                                 static_cast<uint32_t>(eMpiName::Probe), static_cast<uint32_t>(eMpiName::Probe),
	                                 static_cast<uint32_t>(eMpiName::Probe) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_POINT2POINT,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// get count
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Get_count),
	                                 static_cast<uint32_t>(eMpiName::Get_count), static_cast<uint32_t>(eMpiName::Get_count),
	                                 static_cast<uint32_t>(eMpiName::Get_count) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// wait
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Wait),
	                                 static_cast<uint32_t>(eMpiName::Wait), static_cast<uint32_t>(eMpiName::Wait),
	                                 static_cast<uint32_t>(eMpiName::Wait) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// wait all
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Waitall),
	                                 static_cast<uint32_t>(eMpiName::Waitall), static_cast<uint32_t>(eMpiName::Waitall),
	                                 static_cast<uint32_t>(eMpiName::Waitall) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// test some
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Testsome),
	                                 static_cast<uint32_t>(eMpiName::Testsome), static_cast<uint32_t>(eMpiName::Testsome),
	                                 static_cast<uint32_t>(eMpiName::Testsome) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// wait some
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(eMpiName::Waitsome),
	                                 static_cast<uint32_t>(eMpiName::Waitsome), static_cast<uint32_t>(eMpiName::Waitsome),
	                                 static_cast<uint32_t>(eMpiName::Waitsome) + m_id.m_desc_offset,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_MPI,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_mpi_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
	// Pour les points d'entree
	for (const auto& i : m_id.m_ep_id_set) {
		OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, static_cast<uint32_t>(i.m_id),
		                                 static_cast<uint32_t>(i.m_id), static_cast<uint32_t>(i.m_id),
		                                 static_cast<uint32_t>(i.m_id),
		                                 OTF2_REGION_ROLE_FUNCTION,
		                                 OTF2_PARADIGM_NONE,
		                                 OTF2_REGION_FLAG_NONE,
		                                 m_id.m_ep_id, 0, 0);  // 0,0 pour les numeros de lignes des src...
	}
	// synchronize
	OTF2_GlobalDefWriter_WriteRegion(m_global_def_writer, m_id.m_sync_id,
	                                 m_id.m_sync_id, m_id.m_sync_id, m_id.m_sync_id,
	                                 OTF2_REGION_ROLE_FUNCTION,
	                                 OTF2_PARADIGM_NONE,
	                                 OTF2_REGION_FLAG_NONE,
	                                 m_id.m_sync_id, 0, 0);  // 0,0 pour les numeros de lignes du src MPI.c ...
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Internal method to define the regions associated with otf2 events
//! Must be called at the end of archiving
//! This method actually writes the info into the definition file
void Otf2LibWrapper::
_buildOtf2GroupAndCommDefinition()
{
	// Seul le sous-domaine responsable ecrit l'archive otf2
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	// Definitions des sous groupes de communication (nous n'en avons pas donc ce sont les memes)
	OTF2_GlobalDefWriter_WriteGroup(m_global_def_writer, 0 /* id */, m_id.m_empty_id /* name */,
	                                OTF2_GROUP_TYPE_COMM_LOCATIONS, OTF2_PARADIGM_MPI, OTF2_GROUP_FLAG_NONE,
	                                m_sub_domain->nbSubDomain(), m_comm_members.data());

	OTF2_GlobalDefWriter_WriteGroup(m_global_def_writer, 1 /* id */, m_id.m_empty_id /* name */,
	                                OTF2_GROUP_TYPE_COMM_GROUP, OTF2_PARADIGM_MPI,
	                                OTF2_GROUP_FLAG_NONE, m_sub_domain->nbSubDomain(), m_comm_members.data());

	OTF2_GlobalDefWriter_WriteGroup(m_global_def_writer, 2 /* id */, m_id.m_empty_id /* name */,
	                                OTF2_GROUP_TYPE_COMM_SELF, OTF2_PARADIGM_MPI,
	                                OTF2_GROUP_FLAG_NONE, 0, NULL);

	// Definition du communicateur associe au groupe des comm
  OTF2_GlobalDefWriter_WriteComm(m_global_def_writer, 0 /* id */, m_id.m_comm_world_id, 1 /* group */,
                                 OTF2_UNDEFINED_COMM /* parent */
#if OTF2_VERSION_MAJOR >= 3
                                 ,0
#endif
                                 );

  OTF2_GlobalDefWriter_WriteComm(m_global_def_writer, 1 /* id */, m_id.m_comm_self_id, 2 /* group */,
                                 OTF2_UNDEFINED_COMM /* parent */
#if OTF2_VERSION_MAJOR >= 3
                                 ,0
#endif
                                 );

	// Fermeture de la definition de l'archive pour enfin ecrire tout ca
	OTF2_Archive_CloseGlobalDefWriter(m_archive, m_global_def_writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}  // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
