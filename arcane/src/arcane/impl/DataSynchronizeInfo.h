// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeInfo.h                                       (C) 2000-2023 */
/*                                                                           */
/* Informations pour synchroniser les données.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZERINFO_H
#define ARCANE_IMPL_DATASYNCHRONIZERINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/impl/IDataSynchronizeImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
 * \brief Informations nécessaires pour synchroniser les entités sur un groupe.
 *
 * Il faut appeler recompute() après avoir ajouté ou modifier les instances
 * de VariableSyncInfo.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeInfo
: private ReferenceCounterImpl
{
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

  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }

  VariableSyncInfo& rankInfo(Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& rankInfo(Int32 i) const { return m_ranks_info[i]; }

  void clear() { m_ranks_info.clear(); }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s) { m_ranks_info.add(s); }
  Int64 shareDisplacement(Int32 index) const { return m_share_displacements_base[index]; }
  Int64 ghostDisplacement(Int32 index) const { return m_ghost_displacements_base[index]; }
  Int64 totalNbGhost() const { return m_total_nb_ghost; }
  Int64 totalNbShare() const { return m_total_nb_share; }

  //! Notifie l'instance que les indices locaux ont changé
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

  //! Notifie l'instance que les valeurs ont changé
  void recompute();

 public:

  void addReference() { ReferenceCounterImpl::addReference(); }
  void removeReference() { ReferenceCounterImpl::removeReference(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: use operator[] instead")
  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: use operator[] instead")
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }

 private:

  UniqueArray<VariableSyncInfo> m_ranks_info;
  //! Déplacement dans le buffer fantôme de chaque rang
  UniqueArray<Int64> m_ghost_displacements_base;
  //! Déplacement dans le buffer partagé de chaque rang
  UniqueArray<Int64> m_share_displacements_base;
  Int64 m_total_nb_ghost = 0;
  Int64 m_total_nb_share = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
