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
#include "arcane/accelerator/RunCommandLaunchInfo.h"

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
  friend class GenericPartitionerIf;

 public:

  explicit GenericPartitionerBase(const RunQueue& queue);

 protected:

  Int32 _nbFirstPart() const;
  SmallSpan<const Int32> _nbParts() const;
  void _allocate();

 protected:

  RunQueue m_queue;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<int, 2> m_device_nb_list1_storage;
  NumArray<Int32, MDDim1> m_host_nb_list1_storage;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un partitionnement d'une liste.
 */
class GenericPartitionerIf
{
 public:

  /*!
   * \brief Effectue le partitionnement d'une liste en deux parties.
   */
  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericPartitionerBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    RunCommand command = makeCommand(queue);
    command << trace_info;
    impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      int* nb_list1_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(nullptr, temp_storage_size,
                                                   input_iter, output_iter, nb_list1_ptr, nb_item,
                                                   select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(s.m_algo_storage.address(), temp_storage_size,
                                                   input_iter, output_iter, nb_list1_ptr, nb_item,
                                                   select_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(&queue);
      int* nb_list1_ptr = nullptr;
      ARCANE_CHECK_HIP(rocprim::partition(nullptr, temp_storage_size, input_iter, output_iter,
                                          nb_list1_ptr, nb_item, select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();

      ARCANE_CHECK_HIP(rocprim::partition(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                          nb_list1_ptr, nb_item, select_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
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
        if (select_lambda(v)) {
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
    launch_info.endExecute();
  }

  /*!
   * \brief Effectue le partitionnement d'une liste en trois parties.
   */
  template <typename Select1Lambda, typename Select2Lambda,
            typename InputIterator, typename FirstOutputIterator,
            typename SecondOutputIterator, typename UnselectedIterator>
  void apply3(GenericPartitionerBase& s, Int32 nb_item,
              InputIterator input_iter,
              FirstOutputIterator first_output_iter,
              SecondOutputIterator second_output_iter,
              UnselectedIterator unselected_iter,
              const Select1Lambda& select1_lambda,
              const Select2Lambda& select2_lambda,
              const TraceInfo& trace_info = TraceInfo())
  {
    RunQueue queue = s.m_queue;
    eExecutionPolicy exec_policy = queue.executionPolicy();
    RunCommand command = makeCommand(queue);
    command << trace_info;
    impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(&queue);
      // Premier appel pour connaitre la taille pour l'allocation
      int* nb_list1_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(nullptr, temp_storage_size,
                                                   input_iter, first_output_iter, second_output_iter,
                                                   unselected_iter, nb_list1_ptr, nb_item,
                                                   select1_lambda, select2_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();
      ARCANE_CHECK_CUDA(::cub::DevicePartition::If(s.m_algo_storage.address(), temp_storage_size,
                                                   input_iter, first_output_iter, second_output_iter,
                                                   unselected_iter, nb_list1_ptr, nb_item,
                                                   select1_lambda, select2_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(&queue);
      int* nb_list1_ptr = nullptr;
      using namespace rocprim;
      ARCANE_CHECK_HIP(::rocprim::partition_three_way(nullptr, temp_storage_size, input_iter, first_output_iter,
                                                      second_output_iter, unselected_iter,
                                                      nb_list1_ptr, nb_item, select1_lambda, select2_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      nb_list1_ptr = s.m_device_nb_list1_storage.allocate();

      ARCANE_CHECK_HIP(partition_three_way(s.m_algo_storage.address(), temp_storage_size, input_iter, first_output_iter,
                                           second_output_iter, unselected_iter, nb_list1_ptr, nb_item,
                                           select1_lambda, select2_lambda, stream));
      s.m_device_nb_list1_storage.copyToAsync(s.m_host_nb_list1_storage, queue);
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      UniqueArray<bool> filter_index(nb_item);
      Int32 nb_first = 0;
      Int32 nb_second = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        auto v = *input_iter;
        bool is_1 = select1_lambda(v);
        bool is_2 = select2_lambda(v);
        if (is_1) {
          *first_output_iter = v;
          ++first_output_iter;
          ++nb_first;
        }
        else {
          if (is_2) {
            *second_output_iter = v;
            ++second_output_iter;
            ++nb_second;
          }
          else {
            *unselected_iter = v;
            ++unselected_iter;
          }
        }
        // Incrémenter l'itérateur à la fin car il est utilisé pour le positionnement
        ++input_iter;
      }
      s.m_host_nb_list1_storage[0] = nb_first;
      s.m_host_nb_list1_storage[1] = nb_second;
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
    launch_info.endExecute();
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
 * \brief Algorithme générique de partitionnement d'une liste.
 *
 * Cette classe fournit des algorithmes pour partitionner une liste en deux
 * ou trois parties selon un critère fixé par l'utilisateur.
 */
class GenericPartitioner
: private impl::GenericPartitionerBase
{
 public:

  explicit GenericPartitioner(const RunQueue& queue)
  : impl::GenericPartitionerBase(queue)
  {
    _allocate();
  }

 public:

  /*!
   * \brief Effectue un partitionnement d'une liste en deux parties.
   *
   * Le nombre de valeurs de la liste est donné par \a nb_value.
   * Les deux fonctions lambda \a select_lambda et \a setter_lambda permettent
   * de partitionner et de positionner les valeurs de la liste.
   *
   * Après exécution, il est possible de récupérer le nombre d'éléments
   * de la première partie de la liste via la méthode \a nbFirstPart().
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerTwoWayIndex
   */
  template <typename SelectLambda, typename SetterLambda>
  void applyWithIndex(Int32 nb_value, const SetterLambda& setter_lambda,
                      const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (nb_value == 0)
      return;
    _setCalled();
    impl::GenericPartitionerBase* base_ptr = this;
    impl::GenericPartitionerIf gf;

    impl::IndexIterator input_iter;
    impl::SetterLambdaIterator<SetterLambda> out(setter_lambda);
    gf.apply(*base_ptr, nb_value, input_iter, out, select_lambda, trace_info);
  }

  /*!
   * \brief Effectue un partitionnement d'une liste en deux parties.
   *
   * Le nombre de valeurs de la liste est donné par \a nb_value.
   * Les valeurs en entrée sont fournies par l'itérateur \a input_iter et
   * les valeurs en sorties par l'itérateur \a output_iterator. La fonction
   * lambda \a select_lambda permet de sélectionner la partition utilisée :
   * si le retour est \a true, la valeur sera dans la première partie de la liste,
   * sinon elle sera dans la seconde. En sortie les valeurs de la deuxième
   * partie sont rangées en ordre inverse de la liste d'entrée.
   *
   * Après exécution, il est possible de récupérer le nombre d'éléments
   * de la première partie de la liste via la méthode \a nbFirstPart().
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerTwoWayIf
   */
  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_value, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    if (nb_value == 0)
      return;
    _setCalled();
    impl::GenericPartitionerBase* base_ptr = this;
    impl::GenericPartitionerIf gf;
    gf.apply(*base_ptr, nb_value, input_iter, output_iter, select_lambda, trace_info);
  }

  /*!
   * \brief Effectue un partitionnement d'une liste en trois parties.
   *
   * Le nombre de valeurs de la liste est donné par \a nb_value.
   * Les deux fonctions lambda \a select1_lambda et \a select2_lambda permettent
   * de partitionner la liste avec l'algorithme suivant:
   * - si select1_lambda() est vrai, la valeur sera positionnée via \a setter1_lambda,
   * - sinon si select2_lambda() est vrai, la valeur sera positionnée via \a setter2_lambda,
   * - sinon la valeur sera positionnée via \a unselected_setter_lambda.
   *
   * Les listes en sortie sont dans le même ordre qu'en entrée.
   *
   * Après exécution, il est possible de récupérer le nombre d'éléments
   * de la première et de la deuxième liste la méthode \a nbParts().
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerThreeWayIndex
   */
  template <typename Setter1Lambda, typename Setter2Lambda, typename UnselectedSetterLambda,
            typename Select1Lambda, typename Select2Lambda>
  void applyWithIndex(Int32 nb_value,
                      const Setter1Lambda setter1_lambda,
                      const Setter2Lambda setter2_lambda,
                      const UnselectedSetterLambda& unselected_setter_lambda,
                      const Select1Lambda& select1_lambda,
                      const Select2Lambda& select2_lambda,
                      const TraceInfo& trace_info = TraceInfo())
  {
    if (nb_value == 0)
      return;
    _setCalled();
    impl::GenericPartitionerBase* base_ptr = this;
    impl::GenericPartitionerIf gf;
    impl::IndexIterator input_iter;
    impl::SetterLambdaIterator<Setter1Lambda> setter1_wrapper(setter1_lambda);
    impl::SetterLambdaIterator<Setter2Lambda> setter2_wrapper(setter2_lambda);
    impl::SetterLambdaIterator<UnselectedSetterLambda> unselected_setter_wrapper(unselected_setter_lambda);
    gf.apply3(*base_ptr, nb_value, input_iter, setter1_wrapper, setter2_wrapper,
              unselected_setter_wrapper, select1_lambda, select2_lambda, trace_info);
  }

  /*!
   * \brief Effectue un partitionnement d'une liste en trois parties.
   *
   * Le nombre de valeurs de la liste est donné par \a nb_value.
   * Les deux fonctions lambda \a select1_lambda et \a select2_lambda permettent
   * de partitionner la liste avec l'algorithme suivant:
   * - si select1_lambda() est vrai, la valeur ajoutée \a first_output_iter,
   * - sinon si select2_lambda() est vrai, la valeur sera ajoutée à \a second_output_iter,
   * - sinon la valeur sera ajoutée à \a unselected_iter.
   *
   * Les listes en sortie sont dans le même ordre qu'en entrée.
   *
   * Après exécution, il est possible de récupérer le nombre d'éléments
   * de la première et de la deuxième liste la méthode \a nbParts().
   *
   * \snippet AcceleratorPartitionerUnitTest.cc SampleListPartitionerThreeWayIf
   */
  template <typename InputIterator, typename FirstOutputIterator,
            typename SecondOutputIterator, typename UnselectedIterator,
            typename Select1Lambda, typename Select2Lambda>
  void applyIf(Int32 nb_value, InputIterator input_iter,
               FirstOutputIterator first_output_iter,
               SecondOutputIterator second_output_iter,
               UnselectedIterator unselected_iter,
               const Select1Lambda& select1_lambda,
               const Select2Lambda& select2_lambda,
               const TraceInfo& trace_info = TraceInfo())
  {
    if (nb_value == 0)
      return;
    _setCalled();
    impl::GenericPartitionerBase* base_ptr = this;
    impl::GenericPartitionerIf gf;
    gf.apply3(*base_ptr, nb_value, input_iter, first_output_iter, second_output_iter,
              unselected_iter, select1_lambda, select2_lambda, trace_info);
  }

  /*!
   * \brief Nombre d'éléments de la première partie de la liste.
   */
  Int32 nbFirstPart()
  {
    m_is_already_called = false;
    return _nbFirstPart();
  }

  /*!
   * \brief Nombre d'éléments de la première et deuxième partie de la liste.
   *
   * Retourne une vue de deux valeurs. La première valeur contient le nombre
   * d'éléments de la première liste et la seconde valeur le
   * nombre d'éléments de la deuxième liste.
   *
   * Cette méthode n'est valide qu'après avoir appelé une méthode de partitionnement
   * en trois parties.
   */
  SmallSpan<const Int32> nbParts()
  {
    m_is_already_called = false;
    return _nbParts();
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
