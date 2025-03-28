// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2LibWrapper.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Classe qui encapsule les fonctions utiles de la lib Otf2.                 */
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
//! Constructeur.
Otf2LibWrapper::
Otf2LibWrapper(ISubDomain* sub_domain)
: m_sub_domain(sub_domain)
{
	m_flush_callbacks.otf2_pre_flush = _preFlush;
	m_flush_callbacks.otf2_post_flush = _postFlush;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Destructeur.
Otf2LibWrapper::
~Otf2LibWrapper()
{
	// Fermeture de l'archive
	if (m_archive)
	  OTF2_Archive_Close(m_archive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Methode d'initialisation. Permet de definir le chemin ou va se trouver l'archive ainsi que son nom
void Otf2LibWrapper::
init(const String& archive_name)
{
	// Alias sur le comm MPI
	MPI_Comm* mpi_comm((MPI_Comm*)m_sub_domain->parallelMng()->getMPICommunicator());
	if (!mpi_comm)
		ARCANE_FATAL("Impossible d'initialiser la librairie Otf2 sans communicateur MPI");

	// Initialisation du tableau des participants aux comm (i.e. les sub-domain id quoi...)
	m_comm_members.resize(m_sub_domain->nbSubDomain());
  std::iota(m_comm_members.begin(), m_comm_members.end(), 0);

	// Verification de l'eventuelle existance des fichiers et suppression le cas echeant
	String dir_name(m_sub_domain->listingDirectory().path() + "/" + archive_name);
	String otf2_name(dir_name + ".otf2");
	String def_name(dir_name + ".def");
	if (m_sub_domain->parallelMng()->isMasterIO() && std::filesystem::exists(dir_name.localstr())) {
		std::filesystem::remove_all(dir_name.localstr());
		std::filesystem::remove_all(otf2_name.localstr());
		std::filesystem::remove_all(def_name.localstr());
	}
	// Synchro avant d'ouvrir
	MPI_Barrier(*mpi_comm);

	// Sauvegarde du temps de debut pour recaler les enregistrements de chaque rang MPI
	s_epoch_start = getTime();

	// Ouverture de l'archive
	m_archive = OTF2_Archive_Open(m_sub_domain->listingDirectory().path().localstr(),
			                          archive_name.localstr(), OTF2_FILEMODE_WRITE,
	                              1024 * 1024 /* event chunk size */,
	                              4 * 1024 * 1024 /* def chunk size */,
	                              OTF2_SUBSTRATE_POSIX, OTF2_COMPRESSION_NONE);
	if (!m_archive)
		ARCANE_FATAL("Impossible de creer l'archive OTF2");

	// Attachement des callbacks
	OTF2_Archive_SetFlushCallbacks(m_archive, &m_flush_callbacks, NULL);
	if (OTF2_MPI_Archive_SetCollectiveCallbacks(m_archive, *mpi_comm, MPI_COMM_NULL) != OTF2_SUCCESS)
		ARCANE_FATAL("Probleme lors du positionnement des callbacks MPI pour la librairie OTF2");

	// Init des fichiers d'event (pas encore crees)
	OTF2_Archive_OpenEvtFiles(m_archive);

	// Init du event writer
	m_evt_writer = OTF2_Archive_GetEvtWriter(m_archive, m_sub_domain->subDomainId());

	// Creation des ids pour les definitions de l'archive
	_createOtf2Ids();

	// DBG test du contenu du set
	/*
	m_sub_domain->traceMng()->info() << "===== EntryPointIdSet =====";
	for (auto i : m_id.m_ep_id_set)
		m_sub_domain->traceMng()->info() << "{" << i.m_name << ", " << i.m_id << "}";
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Methode a appeler pour finaliser la creation de l'archive (i.e. on ne veut plus enregistrer plus d'evenements)
void Otf2LibWrapper::
finalize()
{
	// Alias sur la comm
	MPI_Comm* mpi_comm((MPI_Comm*)m_sub_domain->parallelMng()->getMPICommunicator());

	// Sauvegarde du temps de fin pour recaler les enregistrements de chaque rang MPI
	uint64_t epoch_end(getTime());

	// Fermeture de ce qui est lie aux events
	OTF2_Archive_CloseEvtWriter(m_archive, m_evt_writer);
  OTF2_Archive_CloseEvtFiles(m_archive);

	// Temporaire pour recuperer les timings globaux
	uint64_t sync_epoch;

	// Echanges pour avoir la fenetre temporelle globale
	// Plus petit temps pour le debut
	/*
	MPI_Reduce(&m_epoch_start, &sync_epoch, 1, OTF2_MPI_UINT64_T, MPI_MIN,
			       m_sub_domain->parallelMng()->masterIORank(), *mpi_comm);
	std::swap(m_epoch_start, sync_epoch);
	 */
	// Plus grand temps pour la fin
	MPI_Reduce(&epoch_end, &sync_epoch, 1, OTF2_MPI_UINT64_T, MPI_MAX,
			       m_sub_domain->parallelMng()->masterIORank(), *mpi_comm);
	std::swap(epoch_end, sync_epoch);

	// On finit de creer les definitions pour l'archive
	//_buildOtf2ClockAndStringDefinition(m_epoch_start, epoch_end);
	_buildOtf2ClockAndStringDefinition(0, epoch_end);
	_buildOtf2ParadigmAndSystemDefinition();
	_buildOtf2LocationDefinition();
	_buildOtf2RegionDefinition();
	_buildOtf2GroupAndCommDefinition();  // L'ecriture effective se passe dans cette methode

	// Synchro avant de fermer
	MPI_Barrier(*mpi_comm);

	// Fermeture du fichier archive
	OTF2_Archive_Close(m_archive);
	m_archive = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Accesseur sur le otf2 event writer interne.
//! Incremente un compteur d'evenement interne a chaque appel.
OTF2_EvtWriter* Otf2LibWrapper::
getEventWriter()
{
	++m_evt_nb;
	return m_evt_writer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Helper pour l'id d'un point d'entree via son nom.
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
//! Helper sur le numero de rank MPI
int Otf2LibWrapper::
getMpiRank() const
{
	return m_sub_domain->subDomainId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Helper sur le nombre de rank MPI
int Otf2LibWrapper::
getMpiNbRank() const
{
	return m_sub_domain->nbSubDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Helper sur le nom de l'application
uint32_t Otf2LibWrapper::
getApplicationNameId() const
{
	return m_id.m_app_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Helper sur la chaine de charactere "syncrhonize"
uint32_t Otf2LibWrapper::
getSynchronizeId() const
{
	return m_id.m_sync_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Méthode interne statique pour recuperer le timestamp.
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

//! Méthode interne statique pour positionner la callback a appeler avant l'evenement a enregistrer
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

//! Méthode interne statique pour positionner la callback a appeler apres l'evenement a enregistrer
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
//! Methode interne de creation des identifiants de nommage pour
//! les definitions necessaires a l'archive OTF2.
//! Doit etre appelee a l'init
void Otf2LibWrapper::
_createOtf2Ids()
{
	// Les ids des noms des operations MPI sont cables sur la valeur de l'enum
	// Leur description est decale d'autant
	// On compteur pour incrementer les identifiants
	u_int32_t offset(static_cast<uint32_t>(eMpiName::NameOffset));
	m_id.m_desc_offset = offset;

	// On continu avec le reste, j'ai pas trouve d'idee geniale pour avoir un indice intelligent ...
	offset *= 2;
	// Chaine vide : 2 x eMpiName::NameOffset
	m_id.m_empty_id = offset++;
	// Nom du thread CPU mappe sur le process MPI : 2 x eMpiName::NameOffset + 1
	m_id.m_thread_id = offset++;
	// Nom de la machine : 2 x eMpiName::NameOffset + 2
	m_id.m_hostname_id = offset++;
	// Classe de la machine... pas inspire... : 2 x eMpiName::NameOffset + 3
	m_id.m_class_id = offset++;
	// MPI Communicator : 2 x eMpiName::NameOffset + 4
	m_id.m_comm_world_id = offset++;
	// MPI : 2 x eMpiName::NameOffset + 5
	m_id.m_mpi_id = offset++;
	// Comm id : 2 x eMpiName::NameOffset + 6
	m_id.m_comm_id = offset++;
	// Win id : 2 x eMpiName::NameOffset + 7
	m_id.m_win_id = offset++;
	// MPI_COMM_SELF : 2 x eMpiName::NameOffset + 8
	m_id.m_comm_self_id = offset++;
	// Entry Point : 2 x eMpiName::NameOffset + 9
	m_id.m_ep_id = offset++;
  // Synchronize : 2 x eMpiName::NameOffset + 10
  m_id.m_sync_id = offset++;
	// Application name : 2 x eMpiName::NameOffset + 11
	m_id.m_app_name = offset++;
	// On continu avec les noms des rangs MPI : a partir de 2 x eMpiName::NameOffset + 12
	m_id.m_rank_offset = offset;
	offset += m_sub_domain->nbSubDomain();

  // On termine avec les points d'entree : a partir de 2 x eMpiName::NameOffset + 12 + nbSubDomain()
  for (auto i : m_sub_domain->timeLoopMng()->usedTimeLoopEntryPoints()) {
		// On ajoute la reference du point d'entree dans la table
    // NOTE: L'identifiant utilisé doit être identique à celui dans
    // Otf2MessagePassingProfilingService
		m_id.m_ep_id_set.emplace(EntryPointId(i->fullName(), offset));
		// On incremente pour tout le monde
		offset++;
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Methode interne de creation de l'ensemble des chaines de caractres utilisees dans
//! les definitions necessaires pour l'archive OTF2.
//! Doit etre appelee au debut de l'archivage
void Otf2LibWrapper::
_buildOtf2ClockAndStringDefinition(uint64_t global_start_time, uint64_t global_end_time)
{
	/*
	 * Seul le sous-domaine responsable des IOs va remplir cette structure OTF2
	 * (parcequ'on ne sait pas trop ce qu'il y a la dessous, mais surement trop de char*)
	 * et pour etre raccord sur les identifiants associes au nom des pts d'entree, on
	 * communique le debut des ids des pts d'entree.
	 */

	// Seul le sous-domaine responsable ecrit l'archive otf2
	if (!m_sub_domain->parallelMng()->isMasterIO())
		return;

	// Recuperation de l'ecrivain
	m_global_def_writer = OTF2_Archive_GetGlobalDefWriter(m_archive);

	// Definition des proprietes temporelles
  // Un nouvel argument est disponible à partir de la version 3.0. Il s'agit du dernier argument
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

	// Definition de toutes les noms que l'on verra apparaitre dans les outils capables de lire l'archive otf2
	// On commence par les noms des operations MPI
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
  // On compteur pour incrementer les identifiants
  u_int32_t offset(static_cast<uint32_t>(eMpiName::NameOffset));

  // On continu avec les descriptions de ces operations. On utilise la valeur du decalage du nb de champ de l'enum
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
//! Methode interne pour definir les regions a associer aux events otf2
//! Doit etre appelee en fin d'archivage
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
//! Methode interne pour ecrire la definition du systeme associee a l'archive otf2
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
//! Methode interne pour definir les regions a associer aux events otf2
//! Doit etre appelee en fin d'archivage
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
//! Methode interne pour definir les regions a associer aux events otf2
//! Doit etre appelee en fin d'archivage
//! C'est cette methode qui ecrit effectivement les infos dans le fichier de def
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
