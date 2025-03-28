// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicParticleExchanger.h                                    (C) 2000-2025 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_BASICPARTICLEEXCHANGER_H
#define ARCANE_MESH_BASICPARTICLEEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IParticleExchanger.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Item.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/FactoryService.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/BasicParticleExchanger_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Echangeur de particules basique (utilise une réduction bloquante).
 */
class BasicParticleExchanger
: public ArcaneBasicParticleExchangerObject
{
 friend class AsyncParticleExchanger;

 public:
  
  explicit BasicParticleExchanger(const ServiceBuildInfo& sbi);
  ~BasicParticleExchanger() override;

 public:

  void build() override {}
  void initialize(IItemFamily* item_family) override;

 public:

  void beginNewExchange(Integer nb_particle) override;
  IItemFamily* itemFamily() override { return m_item_family; }
  bool exchangeItems(Integer nb_particle_finish_exchange,
                     Int32ConstArrayView local_ids,
                     Int32ConstArrayView sub_domains_to_send,ItemGroup item_group,
                     IFunctor* functor) override;
  bool exchangeItems(Integer nb_particle_finish_exchange,
                     Int32ConstArrayView local_ids,
                     Int32ConstArrayView sub_domains_to_send,
                     Int32Array* new_particle_local_ids,
                     IFunctor* functor) override;
  void sendItems(Integer nb_particle_finish_exchange,
                 Int32ConstArrayView local_ids,
                 Int32ConstArrayView sub_domains_to_send) override;
  bool waitMessages(Integer nb_pending_particles,Int32Array* new_particle_local_ids,
                    IFunctor* functor) override;
  void addNewParticles(Integer nb_particle) override;

  void setVerboseLevel(Integer level) override { m_verbose_level = level; }
  Integer verboseLevel() const override { return m_verbose_level; }
  IAsyncParticleExchanger * asyncParticleExchanger() override { return nullptr; }

 public:

  void reset();

 private:

  IItemFamily* m_item_family = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  UniqueArray<ISerializeMessage*> m_accumulate_infos;

  Int32 m_rank = A_NULL_RANK;

  //! Timer
  Timer* m_timer = nullptr;
  Real m_total_time_functor = 0.0;
  Real m_total_time_waiting = 0.0;

  //! Liste des variables à échanger
  VariableList m_variables_to_exchange;
  
  //! Liste des message en attente d'envoie
  UniqueArray<ISerializeMessage*> m_pending_messages;

  //! Liste des message envoyés mais en cours de traitement
  UniqueArray<ISerializeMessage*> m_waiting_messages;

  Ref<ISerializeMessageList> m_message_list;

  Int64 m_nb_total_particle_finish_exchange = 0;

  bool m_exchange_finished = true;
  Integer m_nb_loop = 0;
  bool m_print_info = false;
  Int64 m_last_nb_to_exchange = 0;
  Integer m_current_nb_reduce = 0;
  Integer m_last_nb_reduce = 0;
  Int64 m_nb_particle_send = 0;

  Int32 m_verbose_level = 1;
  Int32 m_debug_exchange_items_level = 0;
  //! Numéro du message. Utile pour le débug
  Int64 m_serialize_id = 1;

  /*!
   * Nombre maximum de messages à envoyer avant de faire la réduction
   * sur le nombre de particules. Si (-1) alors pas de limite.
   */
  Int32 m_max_nb_message_without_reduce = 15;

 private:

  void _clearMessages();
  void _serializeMessage(ISerializeMessage* sm,
                         Int32ConstArrayView acc_ids,
                         Int64Array& items_to_send_uid,
                         Int64Array& items_to_send_cells_uid);
  void _deserializeMessage(ISerializeMessage* message,
                           Int64Array& items_to_create_unique_id,
                           Int64Array& items_to_create_cells_unique_id,
                           Int32Array& items_to_create_local_id,
                           Int32Array& items_to_create_cells_local_id,
                           ItemGroup item_group,
                           Int32Array* new_particle_local_ids);
  void _addItemsToSend(Int32ConstArrayView local_ids,
                       Int32ConstArrayView sub_domains_to_send,
                       Int32ConstArrayView communicating_sub_domains,
                       UniqueArray< SharedArray<Int32> >& ids_to_send);
  void _sendPendingMessages();

  void _generateSendItems(Int32ConstArrayView local_ids,Int32ConstArrayView sub_domains_to_send);
  void _checkInitialized();
  bool _waitMessages(Integer nb_pending_particles,ItemGroup item_group,Int32Array* new_particle_local_ids,IFunctor* functor);
  void _waitMessages(ItemGroup item_group,Int32Array* new_particle_local_ids,IFunctor* functor);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

