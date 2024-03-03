// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Filtering.h                                                 (C) 2000-2024 */
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
#include "arcane/utils/TraceInfo.h"

#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
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
class ARCANE_ACCELERATOR_EXPORT GenericFilteringBase
{
  template <typename DataType, typename FlagType>
  friend class GenericFilteringFlag;
  friend class GenericFilteringIf;

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
class GenericFilteringFlag
{
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
class GenericFilteringIf
{
 public:

  explicit GenericFilteringIf(RunQueue* q)
  : m_queue(q)
  {
  }

 public:

  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericFilteringBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda, const TraceInfo& trace_info)
  {
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue* queue = s.m_queue;
    exec_policy = queue->executionPolicy();
    RunCommand command = makeCommand(*queue);
    command << trace_info;
    impl::RunCommandLaunchInfo launch_info(command, nb_item);
    launch_info.beginExecute();
    switch (exec_policy) {
#if defined(ARCANE_COMPILING_CUDA)
    case eExecutionPolicy::CUDA: {
      size_t temp_storage_size = 0;
      cudaStream_t stream = impl::CudaUtils::toNativeStream(queue);
      // Premier appel pour connaitre la taille pour l'allocation
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::If(nullptr, temp_storage_size,
                                                input_iter, output_iter, nb_out_ptr, nb_item,
                                                select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_out_storage.allocate();
      nb_out_ptr = s.m_device_nb_out_storage.address();
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::If(s.m_algo_storage.address(), temp_storage_size,
                                                input_iter, output_iter, nb_out_ptr, nb_item,
                                                select_lambda, stream));
      ARCANE_CHECK_CUDA(::cudaMemcpyAsync(s.m_host_nb_out_storage.bytes().data(), nb_out_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    } break;
#endif
#if defined(ARCANE_COMPILING_HIP)
    case eExecutionPolicy::HIP: {
      size_t temp_storage_size = 0;
      // Premier appel pour connaitre la taille pour l'allocation
      hipStream_t stream = impl::HipUtils::toNativeStream(queue);
      int* nb_out_ptr = nullptr;
      ARCANE_CHECK_HIP(rocprim::select(nullptr, temp_storage_size, input_iter, output_iter,
                                       nb_out_ptr, nb_item, select_lambda, stream));

      s.m_algo_storage.allocate(temp_storage_size);
      s.m_device_nb_out_storage.allocate();
      nb_out_ptr = s.m_device_nb_out_storage.address();

      ARCANE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                       nb_out_ptr, nb_item, select_lambda, stream));
      ARCANE_CHECK_HIP(::hipMemcpyAsync(s.m_host_nb_out_storage.bytes().data(), nb_out_ptr, sizeof(int), hipMemcpyDeviceToHost, stream));
    }
#endif
    case eExecutionPolicy::Thread:
      // Pas encore implémenté en multi-thread
      [[fallthrough]];
    case eExecutionPolicy::Sequential: {
      Int32 index = 0;
      for (Int32 i = 0; i < nb_item; ++i) {
        if (select_lambda(*input_iter)) {
          *output_iter = *input_iter;
          ++index;
          ++output_iter;
        }
        ++input_iter;
      }
      s.m_host_nb_out_storage[0] = index;
    } break;
    default:
      ARCANE_FATAL(getBadPolicyMessage(exec_policy));
    }
    launch_info.endExecute();
  }

 private:

  RunQueue* m_queue = nullptr;
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

  explicit Filterer(RunQueue* queue)
  {
    m_queue = queue;
    _allocate();
  }

 public:

  /*!
   * \brief Applique un filtre.
   *
   * Filtre tous les éléments de \a input pour lesquels \a flag est différent de 0 et
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
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());

    _setCalled();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringFlag<DataType, FlagType> gf;
    gf.apply(*base_ptr, input, output, flag);
  }

  template <typename SelectLambda>
  ARCANE_DEPRECATED_REASON("Y2024: Use GenericFilterer::applyIf() instead")
  void applyIf(SmallSpan<const DataType> input, SmallSpan<DataType> output,
               const SelectLambda& select_lambda)
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());

    _setCalled();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf(m_queue);
    gf.apply(*base_ptr, nb_item, input.data(), output.data(), select_lambda, TraceInfo());
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
    impl::GenericFilteringFlag<DataType, FlagType> gf;
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

 private:

  void _setCalled()
  {
    if (m_is_deprecated_usage)
      ARCANE_FATAL("You need to create instance with Filterer(RunQueue*) to use this overload of apply()");
    if (m_is_already_called)
      ARCANE_FATAL("apply() has already been called for this instance");
    m_is_already_called = true;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Algorithme générique de filtrage sur accélérateur.
 */
class GenericFilterer
: private impl::GenericFilteringBase
{
 public:

  // NOTE: cette classe devrait être privée mais ce n'est pas possible avec CUDA.
  //! Itérateur sur la lambda via un index
  template <typename SetterLambda>
  class SetterLambdaIterator
  {
   public:

    //! Permet de positionner un élément de l'itérateur de sortie
    class Setter
    {
     public:

      ARCCORE_HOST_DEVICE explicit Setter(const SetterLambda& s, Int32 output_index)
      : m_output_index(output_index)
      , m_lambda(s)
      {}
      ARCCORE_HOST_DEVICE void operator=(Int32 input_index)
      {
        m_lambda(input_index, m_output_index);
      }
      Int32 m_output_index = 0;
      SetterLambda m_lambda;
    };

    using value_type = Setter;
    using iterator_category = std::random_access_iterator_tag;
    using reference = Setter;
    using difference_type = ptrdiff_t;

   public:

    ARCCORE_HOST_DEVICE SetterLambdaIterator(const SetterLambda& s)
    : m_lambda(s)
    {}
    ARCCORE_HOST_DEVICE explicit SetterLambdaIterator(const SetterLambda& s, Int32 v)
    : m_lambda(s)
    , m_index(v)
    {}

   public:

    ARCCORE_HOST_DEVICE SetterLambdaIterator<SetterLambda>& operator++()
    {
      ++m_index;
      return (*this);
    }
    ARCCORE_HOST_DEVICE Setter operator*() const
    {
      return Setter(m_lambda, m_index);
    }
    ARCCORE_HOST_DEVICE value_type operator[](Int32 x) const { return Setter(m_lambda, m_index + x); }

   private:

    Int32 m_index = 0;
    SetterLambda m_lambda;
  };

 public:

  /*!
   * \brief Créé une instance.
   *
   * \pre queue!=nullptr
   */
  explicit GenericFilterer(RunQueue* queue)
  {
    ARCANE_CHECK_POINTER(queue);
    m_queue = queue;
    _allocate();
  }

 public:

  /*!
   * \brief Applique un filtre.
   *
   * Filtre tous les éléments de \a input pour lesquels \a select_lambda vaut \a true et
   * remplit \a output avec les valeurs filtrées. \a output doit avoir une taille assez
   * grande pour contenir tous les éléments filtrés.
   *
   * \a select_lambda doit avoir un opérateur `ARCCORE_HOST_DEVICE bool operator()(const DataType& v) const`.
   *
   * Par exemple la lambda suivante permet de ne garder que les éléments dont
   * la valeur est supérieure à 569.
   *
   * \code
   * auto filter_lambda = [] ARCCORE_HOST_DEVICE (const DataType& x) -> bool {
   *   return (x > 569.0);
   * };
   * \endcode
   *
   * L'algorithme séquentiel est le suivant:
   *
   * \code
   * Int32 index = 0;
   * for (Int32 i = 0; i < nb_item; ++i) {
   *   if (select_lambda(input[i])) {
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
  template <typename DataType, typename SelectLambda>
  void applyIf(SmallSpan<const DataType> input, SmallSpan<DataType> output,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());

    _setCalled();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf(m_queue);
    gf.apply(*base_ptr, nb_item, input.data(), output.data(), select_lambda, trace_info);
  }

  /*!
   * \brief Applique un filtre.
   *
   * Cette méthode est identique à Filterer::applyIf(SmallSpan<const DataType> input,
   * SmallSpan<DataType> output, const SelectLambda& select_lambda) mais permet de spécifier un
   * itérateur \a input_iter pour l'entrée et \a output_iter pour la sortie.
   * Le nombre d'entité en entrée est donné par \a nb_item.
   */
  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _setCalled();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf(m_queue);
    gf.apply(*base_ptr, nb_item, input_iter, output_iter, select_lambda, trace_info);
  }

  /*!
   * \brief Applique un filtre avec une sélection suivant un index.
   *
   * Cette méthode permet de filtrer en spécifiant une fonction lambda à la fois
   * pour la sélection et l'affection. Le prototype de ces lambda est:
   *
   * \code
   * auto select_lambda = [=] ARCCORE_HOST_DEVICE (Int32 index) -> bool;
   * auto setter_lambda = [=] ARCCORE_HOST_DEVICE (Int32 input_index,Int32 output_index) -> void;
   * \endcode
   *
   * Par exemple, si on souhaite recopier dans \a output le tableau \a input dont les
   * valeurs sont supérieures à 25.0.
   *
   * \code
   * SmallSpan<Real> input = ...;
   * SmallSpan<Real> output = ...;
   * auto select_lambda = [=] ARCCORE_HOST_DEVICE (Int32 index) -> bool
   * {
   *   return input[index] > 25.0;
   * };
   * auto setter_lambda = [=] ARCCORE_HOST_DEVICE (Int32 input_index,Int32 output_index)
   * {
   *   output[output_index] = input[input_index];
   * };
   * Arcane::Accelerator::RunQueue* queue = ...;
   * Arcane::Accelerator::GenericFilterer filterer(queue);
   * filterer.applyWithIndex(input.size(), select_lambda, setter_lambda);
   * Int32 nb_out = filterer.nbOutputElement();
   * \endcode
   */
  template <typename SelectLambda, typename SetterLambda>
  void applyWithIndex(Int32 nb_value, const SelectLambda& select_lambda,
                      const SetterLambda& setter_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _setCalled();
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf(m_queue);
    impl::IndexIterator input_iter;
    SetterLambdaIterator<SetterLambda> out(setter_lambda);
    gf.apply(*base_ptr, nb_value, input_iter, out, select_lambda, trace_info);
  }

  //! Nombre d'éléments en sortie.
  Int32 nbOutputElement()
  {
    m_is_already_called = false;
    return _nbOutputElement();
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
