// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeInfo.h                                       (C) 2000-2025 */
/*                                                                           */
/* Informations pour synchroniser les données.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZERINFO_H
#define ARCANE_IMPL_DATASYNCHRONIZERINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class DataSynchronizeInfo;

//! Comparaison des valeurs des entités fantômes avant/après une synchronisation
enum class eDataSynchronizeCompareStatus
{
  //! Pas de comparaison ou résultat inconnue
  Unknown,
  //! Même valeurs avant et après la synchronisation
  Same,
  //! Valeurs différentes avant et après la synchronisation
  Different
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur le résultat d'une synchronisation.
 */
class DataSynchronizeResult
{
 public:

  eDataSynchronizeCompareStatus compareStatus() const { return m_compare_status; }
  void setCompareStatus(eDataSynchronizeCompareStatus v) { m_compare_status = v; }

 private:

  eDataSynchronizeCompareStatus m_compare_status = eDataSynchronizeCompareStatus::Unknown;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la liste des entités partagées/fantômes pour
 * un rang donné pour une synchronisation.
 *
 * TODO: Utiliser pour toutes les VariableSyncInfo un seul tableau pour les
 *       entités partagées et un seul tableau pour les entités fantômes qui
 *       sera géré par ItemGroupSynchronizeInfo.
 */
class ARCANE_IMPL_EXPORT VariableSyncInfo
{
 public:

  VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids, Int32 rank);
  VariableSyncInfo(const VariableSyncInfo& rhs);
  VariableSyncInfo();

 public:

  //! Rang du processeur cible
  Int32 targetRank() const { return m_target_rank; }

  //! localIds() des entités à envoyer au rang targetRank()
  ConstArrayView<Int32> shareIds() const { return m_share_ids; }
  //! localIds() des entités à réceptionner du rang targetRank()
  ConstArrayView<Int32> ghostIds() const { return m_ghost_ids; }

  //! Nombre d'entités partagées
  Int32 nbShare() const { return m_share_ids.size(); }
  //! Nombre d'entités fantômes
  Int32 nbGhost() const { return m_ghost_ids.size(); }

  //! Met à jour les informations lorsque les localId() des entités changent
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

 private:

  //! localIds() des entités à envoyer au processeur #m_rank
  UniqueArray<Int32> m_share_ids;
  //! localIds() des entités à réceptionner du processeur #m_rank
  UniqueArray<Int32> m_ghost_ids;
  //! Rang du processeur cible
  Int32 m_target_rank = A_NULL_RANK;

 private:

  void _changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour les messages d'envoi (share) ou de réception (ghost)
 */
class DataSynchronizeBufferInfoList
{
  friend DataSynchronizeInfo;

 private:

  DataSynchronizeBufferInfoList(const DataSynchronizeInfo* sync_info, bool is_share)
  : m_sync_info(sync_info)
  , m_is_share(is_share)
  {
  }

 public:

  Int32 nbRank() const { return m_displacements_base.size(); }
  //! Nombre total d'éléments
  Int64 totalNbItem() const { return m_total_nb_item; }
  //! Déplacement dans le buffer du rang \a index
  Int64 bufferDisplacement(Int32 index) const { return m_displacements_base[index]; }
  //! Numéros locaux des entités pour le rang \a index
  ConstArrayView<Int32> localIds(Int32 index) const;
  //! Nombre d'entités pour le rang \a index
  Int32 nbItem(Int32 index) const;

 private:

  /*!
   * \brief Offsets dans le buffer global pour chaque rang.
   *
   * Ce tableau est rempli par DataSynchronizeInfo::recompute().
   */
  UniqueArray<Int64> m_displacements_base;
  Int64 m_total_nb_item = 0;
  const DataSynchronizeInfo* m_sync_info = nullptr;
  //! Si vrai, il s'agit du buffer d'envoi, sinon de réception.
  bool m_is_share = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations nécessaires pour synchroniser les entités sur un groupe.
 *
 * Il faut appeler recompute() après avoir ajouté ou modifier les instances
 * de VariableSyncInfo.
 *
 * Les instances de cette classe sont partagées avec tous les dispatchers
 * (IVariableSynchronizeDispatcher) créés à partir d'une instance de
 * IVariableSynchronizer. Seule cette dernière peut donc modifier une instance
 * cette classe.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeInfo
: private ReferenceCounterImpl
{
  friend class DataSynchronizeBufferInfoList;

 private:

  static constexpr int SEND = 0;
  static constexpr int RECEIVE = 1;

 private:

  DataSynchronizeInfo() = default;

 public:

  DataSynchronizeInfo(const DataSynchronizeInfo&) = delete;
  DataSynchronizeInfo operator=(const DataSynchronizeInfo&) = delete;
  DataSynchronizeInfo(DataSynchronizeInfo&&) = delete;
  DataSynchronizeInfo operator=(DataSynchronizeInfo&&) = delete;

 public:

  static Ref<DataSynchronizeInfo> create()
  {
    return makeRef(new DataSynchronizeInfo());
  }

 public:

  void clear() { m_ranks_info.clear(); m_communicating_ranks.clear(); }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s);

  //! Informations d'envoi (partagées)
  const DataSynchronizeBufferInfoList& sendInfo() const { return m_buffer_infos[SEND]; }
  //! Informations de réception (fantômes)
  const DataSynchronizeBufferInfoList& receiveInfo() const { return m_buffer_infos[RECEIVE]; }

  //! Rang de la \a index-ème cible
  Int32 targetRank(Int32 index) const { return m_ranks_info[index].targetRank(); }

  //! Rangs de toutes les cibles
  ConstArrayView<Int32> communicatingRanks() const { return m_communicating_ranks; }

  //! Notifie l'instance que les indices locaux ont changé
  void changeLocalIds(ConstArrayView<Int32> old_to_new_ids);

  //! Notifie l'instance que les valeurs ont changé
  void recompute();

 public:

  void addReference() { ReferenceCounterImpl::addReference(); }
  void removeReference() { ReferenceCounterImpl::removeReference(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }

  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  VariableSyncInfo& rankInfo(Int32 i) { return m_ranks_info[i]; }
  ARCANE_DEPRECATED_REASON("Y2023: do not use")
  const VariableSyncInfo& rankInfo(Int32 i) const { return m_ranks_info[i]; }

 private:

  UniqueArray<Int32> m_communicating_ranks;
  UniqueArray<VariableSyncInfo> m_ranks_info;
  std::array<DataSynchronizeBufferInfoList, 2> m_buffer_infos = { { { this, true }, { this, false } } };

 private:

  DataSynchronizeBufferInfoList& _sendInfo() { return m_buffer_infos[SEND]; }
  DataSynchronizeBufferInfoList& _receiveInfo() { return m_buffer_infos[RECEIVE]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
