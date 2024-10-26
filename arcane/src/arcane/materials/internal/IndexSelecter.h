// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexSelecter.h                                              (C) 2000-2024 */
/*                                                                           */
/* Selection d'index avec API accélérateur                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Filter.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/accelerator/core/Memory.h"

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Construction d'un sous-ensemble d'indexes à partir d'un critère
 *
 */
class IndexSelecter
{
 public:

  IndexSelecter() {}
  IndexSelecter(RunQueue* runqueue)
  // -------------------------------------------------------
  {
    m_is_accelerator_policy = isAcceleratorPolicy(runqueue->executionPolicy());
    m_memory_host = eMemoryRessource(m_is_accelerator_policy ? eMemoryRessource::HostPinned : eMemoryRessource::Host);
    m_memory_device = eMemoryRessource(m_is_accelerator_policy ? eMemoryRessource::Device : eMemoryRessource::Host);
    m_localid_select_device = UniqueArray<Int32>(platform::getDataMemoryRessourceMng()->getAllocator(m_memory_device));
    m_localid_select_host = UniqueArray<Int32>(platform::getDataMemoryRessourceMng()->getAllocator(m_memory_host));
  }

  ~IndexSelecter()
  {
    delete m_generic_filterer_instance;
  }

  /*!
   * \brief Définit l'intervalle [0,nb_idx[ sur lequel va s'opérer la sélection
   */
  void resize(Int32 nb_idx)
  {
    m_index_number = nb_idx;
    m_localid_select_device.resize(m_index_number);
    m_localid_select_host.resize(m_index_number);
  }

  /*!
   * \brief Selectionne les indices selon le prédicat pred et synchronise rqueue_async
   * \return Si host_view, retourne une vue HOST sur les éléments sélectionnés, sinon vue DEVICE
   */
  template <typename PredicateType>
  ConstArrayView<Int32> syncSelectIf(RunQueue* rqueue_async, PredicateType pred, bool host_view = false)
  {
    // On essaie de réutiliser au maximum la même instance de GenericFilterer
    // afin de minimiser des allocations dynamiques dans cette classe.
    // L'instance du GenericFilterer dépend du pointeur de RunQueue donc
    // si ce pointeur change, il faut détruire et réallouer une nouvelle instance.
    bool to_instantiate = (m_generic_filterer_instance == nullptr);
    if (m_asynchronous_queue_pointer != rqueue_async) {
      m_asynchronous_queue_pointer = rqueue_async;
      delete m_generic_filterer_instance;
      to_instantiate = true;
    }
    if (to_instantiate) {
      m_generic_filterer_instance = new GenericFilterer(*m_asynchronous_queue_pointer);
    }

    // On sélectionne dans [0,m_index_number[ les indices i pour lesquels pred(i) est vrai
    //  et on les copie dans out_lid_select.
    //  Le nb d'indices sélectionnés est donné par nbOutputElement()
    SmallSpan<Int32> out_lid_select(m_localid_select_device.data(), m_index_number);

    m_generic_filterer_instance->applyWithIndex(m_index_number, pred,
                                                [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) -> void {
                                                  out_lid_select[output_index] = input_index;
                                                });
    Int32 nb_idx_selected = m_generic_filterer_instance->nbOutputElement();

    if (nb_idx_selected && host_view) {
      // Copie asynchrone Device to Host (m_localid_select_device ==> m_localid_select_host)
      rqueue_async->copyMemory(MemoryCopyArgs(m_localid_select_host.subView(0, nb_idx_selected).data(),
                                              m_localid_select_device.subView(0, nb_idx_selected).data(),
                                              nb_idx_selected * sizeof(Int32))
                               .addAsync());

      rqueue_async->barrier();
    }
    else {
      rqueue_async->barrier();
    }

    ConstArrayView<Int32> lid_select_view = (host_view ? m_localid_select_host.subConstView(0, nb_idx_selected) : m_localid_select_device.subConstView(0, nb_idx_selected));

    return lid_select_view;
  }

 private:

  bool m_is_accelerator_policy = false; // indique si l'accélérateur est disponible ou non
  eMemoryRessource m_memory_host; // identification de l'allocateur HOST
  eMemoryRessource m_memory_device; // identification de l'allocateur DEVICE
  UniqueArray<Int32> m_localid_select_device; // liste des identifiants sélectionnés avec un Filterer (alloué sur DEVICE)
  UniqueArray<Int32> m_localid_select_host; // liste des identifiants sélectionnés avec un Filterer (alloué sur HOST)

  Int32 m_index_number = 0; //!< Intervalle [0, m_index_number[ sur lequel on va opérer la sélection

  RunQueue* m_asynchronous_queue_pointer = nullptr; //!< Pointeur sur la queue du GenericFilterer
  GenericFilterer* m_generic_filterer_instance = nullptr; //!< Instance du GenericFilterer
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
