// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Partitioner.h                                               (C) 2000-2024 */
/*                                                                           */
/* Algorithme de partitionnement de liste.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_PARTITIONER_H
#define ARCANE_ACCELERATOR_PARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NumArray.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/CommonUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour effectuer un filtrage.
 *
 * Contient les arguments nécessaires pour effectuer le filtrage.
 */
class ARCANE_ACCELERATOR_EXPORT GenericPartitionerBase
{
  template <typename DataType, typename FlagType>
  friend class GenericPartitionerFlag;
  friend class GenericPartitionerIf;

 public:

  GenericPartitionerBase();

 protected:

  Int32 _nbFirstPart() const;
  void _allocate();

 protected:

  RunQueue* m_queue = nullptr;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<int> m_device_nb_list1_storage;
  NumArray<Int32, MDDim1> m_host_nb_list1_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un partitionnement d'une liste.
 *
 * La liste est partitionnée en deux listes.
 *
 * \a DataType est le type de donnée.
 * \a FlagType est le type du tableau de filtre.
 */
class GenericPartitionerIf
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericPartitionerBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda)
  {
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue* queue = s.m_queue;
    if (queue)
      exec_policy = queue->executionPolicy();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(queue);
      // Premier appel pour connaitre la taille pour l'allocation
      int* nb_list1_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(nullptr, temp_storage_size,
                                                   input_iter, output_iter, nb_list1_ptr, nb_item,
                                                   select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_list1_storage.allocate();
      nb_list1_ptr = s.m_device_nb_list1_storage.address();
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(s.m_algo_storage.address(), temp_storage_size,
                                                   input_iter, output_iter, nb_list1_ptr, nb_item,
                                                   select_lambda, stream));
      ARCANE_CHECK_CUDA(::cudaMemcpyAsync(s.m_host_nb_list1_storage.bytes().data(), nb_list1_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(queue);
      int* nb_list1_ptr = nullptr;
      ARCANE_CHECK_HIP(rocprim::partition(nullptr, temp_storage_size, input_iter, output_iter,
                                          nb_list1_ptr, nb_item, select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_list1_storage.allocate();
      nb_list1_ptr = s.m_device_nb_list1_storage.address();

      ARCANE_CHECK_HIP(rocprim::partition(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                          nb_list1_ptr, nb_item, select_lambda, stream));
      ARCANE_CHECK_HIP(::hipMemcpyAsync(s.m_host_nb_list1_storage.bytes().data(), nb_list1_ptr,
                                        sizeof(int), hipMemcpyDeviceToHost, stream));
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      UniqueArray<bool> filter_index(nb_item);
      auto saved_output_iter = output_iter;
      auto output2_iter = output_iter + nb_item;
      for (Int32 i = 0; i < nb_item; ++i) {
        auto v = *input_iter;
        if (select_lambda(*input_iter)) {
          *output_iter = v;
          ++output_iter;
        }
        else {
          --output2_iter;
          *output2_iter = v;
        }
        ++input_iter;
      }
      Int32 nb_list1 = static_cast<Int32>(output_iter - saved_output_iter);
      s.m_host_nb_list1_storage[0] = nb_list1;
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme générique de filtrage sur accélérateur.
 *
 * Dans les méthodes suivantes, l'argument \a queue peut être nul auquel cas
 * l'algorithme s'applique sur l'hôte en séquentiel.
 */
class GenericPartitioner
: private impl::GenericPartitionerBase
{
 public:

  explicit GenericPartitioner(RunQueue* queue)
  {
    m_queue = queue;
    _allocate();
  }

 public:

  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda)
  {
    _setCalled();
    impl::GenericPartitionerBase* base_ptr = this;
    impl::GenericPartitionerIf gf;
    gf.apply(*base_ptr, nb_item, input_iter, output_iter, select_lambda);
  }

  //! Nombre d'éléments de la première partie de la liste.
  Int32 nbFirstPart()
  {
    m_is_already_called = false;
    return _nbFirstPart();
  }

 private:

  bool m_is_already_called = false;

 private:

  void _setCalled()
  {
    if (m_is_already_called)
      ARCANE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
