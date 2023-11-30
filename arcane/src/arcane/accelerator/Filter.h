// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Filtering.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Algorithme de filtrage.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_FILTERING_H
#define ARCANE_ACCELERATOR_FILTERING_H
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
class ARCANE_ACCELERATOR_EXPORT GenericFilteringBase
{
  template <typename DataType, typename FlagType>
  friend class GenericFiltering;

 public:

  GenericFilteringBase();

 protected:

  Int32 _nbOutputElement() const;
  void _allocate();

 protected:

  RunQueue* m_queue = nullptr;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<int> m_device_nb_out_storage;
  NumArray<Int32, MDDim1> m_host_nb_out_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un filtrage
 *
 * \a DataType est le type de donnée.
 * \a FlagType est le type du tableau de filtre.
 */
template <typename DataType, typename FlagType>
class GenericFiltering
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  void apply(GenericFilteringBase& s, SmallSpan<const DataType> input, SmallSpan<DataType> output, SmallSpan<const FlagType> flag)
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    [[maybe_unused]] const DataType* input_data = input.data();
    [[maybe_unused]] DataType* output_data = output.data();
    [[maybe_unused]] const FlagType* flag_data = flag.data();
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
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::Flagged(nullptr, temp_storage_size,
                                                     input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_out_storage.allocate();
      nb_out_ptr = s.m_device_nb_out_storage.address();
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::Flagged(s.m_algo_storage.address(), temp_storage_size,
                                                     input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));
      ARCANE_CHECK_CUDA(::cudaMemcpyAsync(s.m_host_nb_out_storage.bytes().data(), nb_out_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(queue);
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_HIP(rocprim::select(nullptr, temp_storage_size, input_data, flag_data, output_data,
                                       nb_out_ptr, nb_item, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_out_storage.allocate();
      nb_out_ptr = s.m_device_nb_out_storage.address();

      ARCANE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_data, flag_data, output_data,
                                       nb_out_ptr, nb_item, stream));
      ARCANE_CHECK_HIP(::hipMemcpyAsync(s.m_host_nb_out_storage.bytes().data(), nb_out_ptr, sizeof(int), hipMemcpyDeviceToHost, stream));
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 index = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        if (flag[i] != 0) {
          output[index] = input[i];
          ++index;
        }
      }
      s.m_host_nb_out_storage[0] = index;
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
  }

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme de filtrage sur accélérateur.
 *
 * Dans les méthodes suivantes, l'argument \a queue peut être nul auquel cas
 * l'algorithme s'applique sur l'hôte en séquentiel.
 *
 * Les instances de cette classe ne peuvent servir qu'une seule fois.
 */
template <typename DataType>
class Filterer
: private impl::GenericFilteringBase
{
 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use Filterer(RunQueue*) instead")
  Filterer()
  : m_is_deprecated_usage(true)
  {
  }

 public:

  Filterer(RunQueue* queue)
  {
    m_queue = queue;
    _allocate();
  }

 public:

  /*!
   * \brief Applique le filtre.
   *
   * Filtre tous les éléments de \a input pour lesquels \a flag vaut 1 et
   * remplit \a output avec les valeurs filtrées. \a output doit avoir une taille assez
   * grande pour contenir tous les éléments filtrés.
   *
   * L'algorithme séquentiel est le suivant:
   *
   * \code
   * Int32 index = 0;
   * for (Int32 i = 0; i < nb_item; ++i) {
   *   if (flag[i] != 0) {
   *     output[index] = input[i];
   *     ++index;
   *   }
   * }
   * return index;
   * \endcode
   *
   * Il faut appeler la méthode nbOutputElement() pour obtenir le nombre d'éléments
   * après filtrage.
   */
  template <typename FlagType>
  void apply(SmallSpan<const DataType> input, SmallSpan<DataType> output, SmallSpan<const FlagType> flag)
  {
    if (m_is_deprecated_usage)
      ARCANE_FATAL("You need to create instance with Filterer(RunQueue*) to use this overload of apply()");
    if (m_is_already_called)
      ARCANE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFiltering<DataType, FlagType> gf;
    gf.apply(*base_ptr, input, output, flag);
  }

  template <typename FlagType>
  ARCANE_DEPRECATED_REASON("Y2023: Use apply() without RunQueue argument instead")
  void apply(RunQueue* queue, SmallSpan<const DataType> input, SmallSpan<DataType> output, SmallSpan<const FlagType> flag)
  {
    if (!m_is_deprecated_usage)
      ARCANE_FATAL("This overload of apply() is only valid when using default constructor");
    if (m_is_already_called)
      ARCANE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
    m_queue = queue;
    _allocate();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFiltering<DataType, FlagType> gf;
    gf.apply(*base_ptr, input, output, flag);
  }

  //! Nombre d'éléments en sortie.
  Int32 nbOutputElement()
  {
    m_is_already_called = false;
    return _nbOutputElement();
  }

 private:

  bool m_is_already_called = false;
  bool m_is_deprecated_usage = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
