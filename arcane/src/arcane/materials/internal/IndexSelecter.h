// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexSelecter.h                                             (C) 2000-2024 */
/*                                                                           */
/* Index selection with accelerator API                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryUtils.h"

#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/GenericFilterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Construction of a subset of indices from a criterion
 *
 */
class IndexSelecter
{
 public:

  IndexSelecter() {}
  IndexSelecter(const RunQueue& runqueue)
  // -------------------------------------------------------
  {
    m_is_accelerator_policy = isAcceleratorPolicy(runqueue.executionPolicy());
    m_memory_host = eMemoryRessource(m_is_accelerator_policy ? eMemoryRessource::HostPinned : eMemoryRessource::Host);
    m_memory_device = eMemoryRessource(m_is_accelerator_policy ? eMemoryRessource::Device : eMemoryRessource::Host);
    m_localid_select_device = UniqueArray<Int32>(MemoryUtils::getAllocator(m_memory_device));
    m_localid_select_host = UniqueArray<Int32>(MemoryUtils::getAllocator(m_memory_host));
  }

  ~IndexSelecter()
  {
    delete m_generic_filterer_instance;
  }

  /*!
   * \brief Defines the interval [0,nb_idx[ on which the selection will be performed
   */
  void resize(Int32 nb_idx)
  {
    m_index_number = nb_idx;
    m_localid_select_device.resize(m_index_number);
    m_localid_select_host.resize(m_index_number);
  }

  /*!
   * \brief Selects the indices according to the predicate pred and synchronizes rqueue_async
   * \return If host_view, returns a HOST view of the selected elements, otherwise a DEVICE view
   */
  template <typename PredicateType>
  ConstArrayView<Int32> syncSelectIf(const RunQueue& rqueue_async, PredicateType pred, bool host_view = false)
  {
    // We try to reuse the same GenericFilterer instance as much as possible
    // in order to minimize dynamic allocations in this class.
    // The GenericFilterer instance depends on the RunQueue pointer, so
    // if this pointer changes, a new instance must be destroyed and reallocated.
    bool to_instantiate = (m_generic_filterer_instance == nullptr);
    if (m_asynchronous_queue_pointer != rqueue_async) {
      m_asynchronous_queue_pointer = rqueue_async;
      delete m_generic_filterer_instance;
      to_instantiate = true;
    }
    if (to_instantiate) {
      m_generic_filterer_instance = new GenericFilterer(m_asynchronous_queue_pointer);
    }

    // We select the indices i in [0,m_index_number[ for which pred(i) is true
    // and copy them into out_lid_select.
    // The number of selected indices is given by nbOutputElement()
    SmallSpan<Int32> out_lid_select(m_localid_select_device.data(), m_index_number);

    m_generic_filterer_instance->applyWithIndex(m_index_number, pred,
                                                [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) -> void {
                                                  out_lid_select[output_index] = input_index;
                                                });
    Int32 nb_idx_selected = m_generic_filterer_instance->nbOutputElement();

    if (nb_idx_selected && host_view) {
      // Asynchronous copy Device to Host (m_localid_select_device ==> m_localid_select_host)
      rqueue_async.copyMemory(MemoryCopyArgs(m_localid_select_host.subView(0, nb_idx_selected).data(),
                                             m_localid_select_device.subView(0, nb_idx_selected).data(),
                                             nb_idx_selected * sizeof(Int32))
                              .addAsync());
    }
    rqueue_async.barrier();

    ConstArrayView<Int32> lid_select_view = (host_view ? m_localid_select_host.subConstView(0, nb_idx_selected) : m_localid_select_device.subConstView(0, nb_idx_selected));

    return lid_select_view;
  }

 private:

  bool m_is_accelerator_policy = false; // indicates whether the accelerator is available or not
  eMemoryRessource m_memory_host; // identification of the HOST allocator
  eMemoryRessource m_memory_device; // identification of the DEVICE allocator
  UniqueArray<Int32> m_localid_select_device; // list of selected IDs using a Filterer (allocated on DEVICE)
  UniqueArray<Int32> m_localid_select_host; // list of selected IDs using a Filterer (allocated on HOST)

  Int32 m_index_number = 0; //!< Interval [0, m_index_number[ on which the selection will be performed

  RunQueue m_asynchronous_queue_pointer; //!< Pointer to the GenericFilterer queue
  GenericFilterer* m_generic_filterer_instance = nullptr; //!< GenericFilterer instance
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
