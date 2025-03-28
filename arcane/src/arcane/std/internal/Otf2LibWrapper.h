// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Otf2LibWrapper.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classe qui encapsule les fonctions utiles de la lib Otf2.                 */
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
 * \brief Classe d'encapsulation des fonctions de la librairie OTF2.
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

	static OTF2_FlushType _preFlush(void* user_data,OTF2_FileType file_type, OTF2_LocationRef location, void* caller_data, bool final);
	static OTF2_TimeStamp _postFlush(void* user_data, OTF2_FileType file_type, OTF2_LocationRef location);

 public:
  // Une structure simple pour associer un nom de point d'entree avec un identifiant
	struct EntryPointId {
		EntryPointId(String name, Integer id) : m_name(name), m_id(id) {}
		String m_name;
		Integer m_id;
		// Foncteur pour pouvoir faire un "heterogeneous lookup" sur le nom pour l'id
		struct EntryPointIdCompare {
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
  // Structure interne pour la gestion des indices et decalages...
	struct InternalIds {
    uint32_t m_desc_offset;   // decalage des description pour les noms des op MPI
		uint32_t m_empty_id;      // id pour la string vide
		uint32_t m_thread_id;     // id pour le nom du thread CPU mappe sur le process MPI
    uint32_t m_hostname_id;   // id pour le nom de la machine
		uint32_t m_class_id;      // id pour la classe de la machine
		uint32_t m_comm_world_id; // id pour le MPI_COMM_WOLRD
		uint32_t m_mpi_id;        // id pour la string MPI
		uint32_t m_comm_id;       // id pour la string du paradigm MPI
		uint32_t m_win_id;        // id pour la string du paradigm MPI
		uint32_t m_comm_self_id;  // id pour la string du comme self
		uint32_t m_ep_id;         // id pour la string Entry Point
		uint32_t m_sync_id;       // id pour la string Synchronize
		uint32_t m_app_name;      // id pour le nom de l'application
		uint32_t m_rank_offset;   // decalage pour les ranks MPI
		std::set<EntryPointId, EntryPointId::EntryPointIdCompare> m_ep_id_set;
		// Ensemble des noms des points d'entree et leur id
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

}  // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
