// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NonBlockingParticleExchanger.h                              (C) 2000-2020 */
/*                                                                           */
/* Echangeur de particules.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_NONBLOCKINGPARTICLEEXCHANGER_H
#define ARCANE_NONBLOCKINGPARTICLEEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/IParticleExchanger.h"
#include "arcane/BasicService.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;
class SerializeMessage;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Echangeur de particules.
 */
class NonBlockingParticleExchanger
: public BasicService
, public IParticleExchanger
{
 private:
  static const Integer MESSAGE_EXCHANGE = 1;
  static const Integer MESSAGE_NB_FINISH_EXCHANGE = 2;
  static const Integer MESSAGE_FINISH_EXCHANGE_STATUS = 3;
  static const Integer MESSAGE_CHANGE_BLOCKING = 4;
 public:
  
  explicit NonBlockingParticleExchanger(const ServiceBuildInfo& sbi);
  ~NonBlockingParticleExchanger() override;

 public:

  void build() override {}
  void initialize(IItemFamily* item_family) override;

 public:

  void beginNewExchange(Integer nb_particule) override;
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

  bool waitMessages(Integer nb_pending_particle,Int32Array* new_particle_local_ids,
                    IFunctor* functor) override;
  void addNewParticles(Integer nb_particle) override
  {
    ARCANE_UNUSED(nb_particle);
    throw NotImplementedException(A_FUNCINFO);
  }
  void setVerboseLevel(Integer level) override { m_verbose_level = level; }
  Integer verboseLevel() const override { return m_verbose_level; }
  IAsyncParticleExchanger * asyncParticleExchanger() override { return nullptr; }

  void reset();

 private:

  IItemFamily* m_item_family;
  IParallelMng* m_parallel_mng;
  UniqueArray<SerializeMessage*> m_accumulate_infos;

  Int32 m_rank;

  //! Timer
  Timer* m_timer;
  Real m_total_time_functor;
  Real m_total_time_waiting;

  //! Liste des variables à échanger
  VariableList m_variables_to_exchange;
  
  //! Liste des message en attente d'envoie
  UniqueArray<ISerializeMessage*> m_pending_messages;

  //! Liste des message envoyés mais en cours de traitement
  UniqueArray<ISerializeMessage*> m_waiting_messages;

  Ref<ISerializeMessageList> m_message_list;

  Int64 m_nb_total_particle_finish_exchange;
  Int64 m_nb_total_particle;

  Integer m_nb_original_blocking_size;
  //! Nombre de particules restantes avant de passer en mode bloquant.
  Integer m_nb_blocking_size;

  bool m_exchange_finished;
  Int32 m_master_proc;
  bool m_need_general_receive;
  bool m_end_message_sended;
  bool m_can_process_messages;
  bool m_can_process_non_blocking;

  bool m_want_process_non_blocking;
  bool m_want_fast_send_particles;

  Integer m_nb_receive_message;

  Int64 m_nb_particle_finished_exchange;

  Int32UniqueArray m_waiting_local_ids;
  Int32UniqueArray m_waiting_sub_domains_to_send;
  Integer m_verbose_level;
  bool m_is_debug;

 private:

  void _clearMessages();
  void _serializeMessage(ISerializeMessage* msg,
                         Int32ConstArrayView acc_ids,
                         Int64Array& items_to_send_uid,
                         Int64Array& items_to_send_cells_uid);
  void _deserializeMessage(ISerializeMessage* msg,
                           Int64Array& items_to_create_id,
                           Int64Array& items_to_create_cell_id,
                           ItemGroup item_group,
                           Int32Array* new_particle_local_ids);
  void _processFinishTrackingMessage();
  void _addFinishExchangeParticle(Int64 nb_particle_finish_exchange);
  void _sendFinishExchangeParticle();
  void _addItemsToSend(Int32ConstArrayView local_ids,
                       Int32ConstArrayView sub_domains_to_send,
                       Int32ConstArrayView communicating_sub_domains,
                       UniqueArray<SharedArray<Int32> >& ids_to_send);
  void _sendPendingMessages();
  void _checkNeedReceiveMessage();
  bool _exchangeItems(Int32ConstArrayView local_ids,
                      Int32ConstArrayView sub_domains_to_send,
                      ItemGroup item_group,
                      Int32Array* new_particle_local_ids,
                      IFunctor* functor);
  void _checkSendItems(Int32ConstArrayView local_ids,Int32ConstArrayView sub_domains_to_send);
  void _generateSendItemsMessages(Int32ConstArrayView local_ids,Int32ConstArrayView sub_domains_to_send);
  void _checkInitialized();
  void _processMessages(ItemGroup item_group,Int32Array* new_particle_local_ids,bool wait_all,IFunctor* functor);
  bool _waitMessages(Integer nb_pending_particle,ItemGroup group,Int32Array* new_particle_local_ids,IFunctor* functor);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
