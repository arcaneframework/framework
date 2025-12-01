// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GenericReducer.h                                            (C) 2000-2025 */
/*                                                                           */
/* Gestion des réductions pour les accélérateurs.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_GENERICREDUCER_H
#define ARCCORE_ACCELERATOR_GENERICREDUCER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/NumArray.h"
#include "arccore/common/accelerator/RunQueue.h"

#include "arccore/accelerator/Reduce.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
template <typename DataType>
class GenericReducerIf;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Classe pour déterminer l'instance de 'Reducer2' à utiliser en fonction de l'opérateur.
// A spécialiser.
template <typename DataType, typename Operator>
class ReduceOperatorToReducerTypeTraits;

template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, MaxOperator<DataType>>
{
 public:

  using ReducerType = ReducerMax2<DataType>;
};
template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, MinOperator<DataType>>
{
 public:

  using ReducerType = ReducerMin2<DataType>;
};
template <typename DataType>
class ReduceOperatorToReducerTypeTraits<DataType, SumOperator<DataType>>
{
 public:

  using ReducerType = ReducerSum2<DataType>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour effectuer une réduction.
 *
 * Contient les arguments nécessaires pour effectuer une réduction.
 */
template <typename DataType>
class GenericReducerBase
{
  friend class GenericReducerIf<DataType>;

 public:

  GenericReducerBase(const RunQueue& queue)
  : m_queue(queue)
  {}

 protected:

  DataType _reducedValue() const
  {
    m_queue.barrier();
    return m_host_reduce_storage[0];
  }

  void _allocate()
  {
    eMemoryResource r = eMemoryResource::HostPinned;
    if (m_host_reduce_storage.memoryRessource() != r)
      m_host_reduce_storage = NumArray<DataType, MDDim1>(r);
    m_host_reduce_storage.resize(1);
  }

 protected:

  RunQueue m_queue;
  GenericDeviceStorage m_algo_storage;
  DeviceStorage<DataType> m_device_reduce_storage;
  NumArray<DataType, MDDim1> m_host_reduce_storage;
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
 */
template <typename DataType>
class GenericReducerIf
{
  // TODO: Faire le malloc sur le device associé à la queue.
  //       et aussi regarder si on peut utiliser mallocAsync().

 public:

  template <typename InputIterator, typename ReduceOperator>
  void apply(GenericReducerBase<DataType>& s, Int32 nb_item, const DataType& init_value,
             InputIterator input_iter, ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    RunQueue& queue = s.m_queue;
    RunCommand command = makeCommand(queue);
    command << trace_info;
    Impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    eExecutionPolicy exec_policy = queue.executionPolicy();
    switch (exec_policy) {
#if defined(ARCCORE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = Impl::CudaUtils::toNativeStream(queue);
      DataType* reduced_value_ptr = nullptr;
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_CUDA(::cub::DeviceReduce::Reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr,
                                                    nb_item, reduce_op, init_value, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();
      ARCANE_CHECK_CUDA(::cub::DeviceReduce::Reduce(s.m_algo_storage.address(), temp_storage_size,
                                                    input_iter, reduced_value_ptr, nb_item,
                                                    reduce_op, init_value, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      hipStream_t stream = Impl::HipUtils::toNativeStream(queue);
      DataType* reduced_value_ptr = nullptr;
      // Premier appel pour connaitre la taille pour l'allocation
      ARCANE_CHECK_HIP(rocprim::reduce(nullptr, temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                       nb_item, reduce_op, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      reduced_value_ptr = s.m_device_reduce_storage.allocate();

      ARCANE_CHECK_HIP(rocprim::reduce(s.m_algo_storage.address(), temp_storage_size, input_iter, reduced_value_ptr, init_value,
                                       nb_item, reduce_op, stream));
      s.m_device_reduce_storage.copyToAsync(s.m_host_reduce_storage, queue);
    } break;
#endif
#if defined(ARCCORE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      {
        RunCommand command2 = makeCommand(queue);
        using ReducerType = typename ReduceOperatorToReducerTypeTraits<DataType, ReduceOperator>::ReducerType;
        ReducerType reducer(command2);
        command2 << RUNCOMMAND_LOOP1(iter, nb_item, reducer)
        {
          auto [i] = iter();
          reducer.combine(input_iter[i]);
        };
        queue.barrier();
        s.m_host_reduce_storage[0] = reducer.reducedValue();
      }
    } break;
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      DataType reduced_value = init_value;
      for (Int32 i = 0; i < nb_item; ++i) {
        reduced_value = reduce_op(reduced_value, *input_iter);
        ++input_iter;
      }
      s.m_host_reduce_storage[0] = reduced_value;
    } break;
    default:
      ARCCORE_FATAL(getBadPolicyMessage(exec_policy));
    }
    launch_info.endExecute();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme générique de réduction sur accélérateur.
 *
 * La réduction se fait via les appels à applyMin(), applyMax(), applySum(),
 * applyMinWithIndex(), applyMaxWithIndex() ou applySumWithIndex(). Ces
 * méthodes sont asynchrones.  Après réduction, il est possible récupérer la
 * valeur réduite via reducedValue(). L'appel à reducedValue() bloque tant
 * que la réduction n'est pas terminée.
 *
 * Les instances de cette classe peuvent être utilisées plusieurs fois.
 *
 * Voici un exemple pour calculer la somme d'un tableau de 50 éléments:
 *
 * \code
 * using namespace Arcane;
 * const Int32 nb_value(50);
 * Arcane::NumArray<Real, MDDim1> t1(nv_value);
 * Arcane::SmallSpan<const Real> t1_view(t1);
 * Arcane::RunQueue queue = ...;
 * Arcane::Accelerator::GenericReducer<Real> reducer(queue);
 *
 * // Calcul direct
 * reducer.applySum(t1_view);
 * std::cout << "Sum is '" << reducer.reducedValue() << "\n";
 *
 * // Calcul avec lambda
 * auto getter_func = [=] ARCCORE_HOST_DEVICE(Int32 index) -> Real
 * {
 *   return t1_view[index];
 * }
 * reducer.applySumWithIndex(nb_value,getter_func);
 * std::cout << "Sum is '" << reducer.reducedValue() << "\n";
 * \endcode
 */
template <typename DataType>
class GenericReducer
: private impl::GenericReducerBase<DataType>
{
 public:

  explicit GenericReducer(const RunQueue& queue)
  : impl::GenericReducerBase<DataType>(queue)
  {
    this->_allocate();
  }

 public:

  //! Applique une réduction 'Min' sur les valeurs \a values
  void applyMin(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), impl::MinOperator<DataType>{}, trace_info);
  }

  //! Applique une réduction 'Max' sur les valeurs \a values
  void applyMax(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), impl::MaxOperator<DataType>{}, trace_info);
  }

  //! Applique une réduction 'Somme' sur les valeurs \a values
  void applySum(SmallSpan<const DataType> values, const TraceInfo& trace_info = TraceInfo())
  {
    _apply(values.size(), values.data(), impl::SumOperator<DataType>{}, trace_info);
  }

  //! Applique une réduction 'Min' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applyMinWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, impl::MinOperator<DataType>{}, trace_info);
  }

  //! Applique une réduction 'Max' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applyMaxWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, impl::MaxOperator<DataType>{}, trace_info);
  }

  //! Applique une réduction 'Somme' sur les valeurs sélectionnées par \a select_lambda
  template <typename SelectLambda>
  void applySumWithIndex(Int32 nb_value, const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _applyWithIndex(nb_value, select_lambda, impl::SumOperator<DataType>{}, trace_info);
  }

  //! Valeur de la réduction
  DataType reducedValue()
  {
    m_is_already_called = false;
    return this->_reducedValue();
  }

 private:

  bool m_is_already_called = false;

 private:

  template <typename InputIterator, typename ReduceOperator>
  void _apply(Int32 nb_value, InputIterator input_iter, ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    _setCalled();
    impl::GenericReducerBase<DataType>* base_ptr = this;
    impl::GenericReducerIf<DataType> gf;
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op, trace_info);
  }

  template <typename GetterLambda, typename ReduceOperator>
  void _applyWithIndex(Int32 nb_value, const GetterLambda& getter_lambda,
                       ReduceOperator reduce_op, const TraceInfo& trace_info)
  {
    _setCalled();
    impl::GenericReducerBase<DataType>* base_ptr = this;
    impl::GenericReducerIf<DataType> gf;
    impl::GetterLambdaIterator<DataType, GetterLambda> input_iter(getter_lambda);
    DataType init_value = reduce_op.defaultValue();
    gf.apply(*base_ptr, nb_value, init_value, input_iter, reduce_op, trace_info);
  }

  void _setCalled()
  {
    if (m_is_already_called)
      ARCCORE_FATAL("apply() has already been called for this instance");
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
