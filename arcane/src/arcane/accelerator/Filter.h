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
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/ScanImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
//#define ARCANE_USE_SCAN_ONEDPL

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
  template <typename DataType, typename FlagType, typename OutputDataType>
  friend class GenericFilteringFlag;
  friend class GenericFilteringIf;
  friend class SyclGenericFilteringImpl;

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

    using value_type = Int32;
    using iterator_category = std::random_access_iterator_tag;
    using reference = Setter;
    using difference_type = ptrdiff_t;
    using pointer = void;

    using ThatClass = SetterLambdaIterator<SetterLambda>;

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
    ARCCORE_HOST_DEVICE reference operator*() const
    {
      return Setter(m_lambda, m_index);
    }
    ARCCORE_HOST_DEVICE reference operator[](Int32 x) const { return Setter(m_lambda, m_index + x); }
    ARCCORE_HOST_DEVICE friend ThatClass operator+(Int32 x, const ThatClass& iter)
    {
      return ThatClass(iter.m_lambda, iter.m_index + x);
    }
    ARCCORE_HOST_DEVICE friend ThatClass operator+(const ThatClass& iter, Int32 x)
    {
      return ThatClass(iter.m_lambda, iter.m_index + x);
    }
    ARCCORE_HOST_DEVICE Int32 operator-(const ThatClass& x) const
    {
      return m_index - x.m_index;
    }
    ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& iter1, const ThatClass& iter2)
    {
      return iter1.m_index < iter2.m_index;
    }

   private:

    Int32 m_index = 0;
    SetterLambda m_lambda;
  };

 protected:

  GenericFilteringBase();

 protected:

  Int32 _nbOutputElement() const;
  void _allocate();
  void _allocateTemporaryStorage(size_t size);
  int* _getDeviceNbOutPointer();
  void _copyDeviceNbOutToHostNbOut();

 protected:

  //! File d'exécution. Ne doit pas être nulle.
  RunQueue* m_queue = nullptr;
  // Mémoire de travail pour l'algorithme de filtrage.
  GenericDeviceStorage m_algo_storage;
  //! Mémoire sur le device du nombre de valeurs filtrées
  DeviceStorage<int> m_device_nb_out_storage;
  //! Mémoire hôte pour le nombre de valeurs filtrées
  NumArray<Int32, MDDim1> m_host_nb_out_storage;
  /*!
   * \brief Indique quelle mémoire est utilisée pour le nombre de valeurs filtrées.
   *
   * Si vrai utilise directement \a m_host_nb_out_storage. Sinon, utilise
   * m_device_nb_out_storage et fait une copie asynchrone après le filtrage pour
   * recopier la valeur dans m_host_nb_out_storage.
   */
  bool m_use_direct_host_storage = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)
//! Implémentation pour SYCL
class SyclGenericFilteringImpl
{
 public:

  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  static void apply(GenericFilteringBase& s, Int32 nb_item, InputIterator input_iter,
                    OutputIterator output_iter, SelectLambda select_lambda)
  {
    RunQueue* queue = s.m_queue;
    using DataType = std::iterator_traits<OutputIterator>::value_type;
#if defined(ARCANE_USE_SCAN_ONEDPL) && defined(__INTEL_LLVM_COMPILER)
    sycl::queue true_queue = impl::SyclUtils::toNativeStream(queue);
    auto policy = oneapi::dpl::execution::make_device_policy(true_queue);
    auto out_iter = oneapi::dpl::copy_if(policy, input_iter, input_iter + nb_item, output_iter, select_lambda);
    Int32 nb_output = out_iter - output_iter;
    s.m_host_nb_out_storage[0] = nb_output;
#else
    NumArray<Int32, MDDim1> scan_input_data(nb_item);
    NumArray<Int32, MDDim1> scan_output_data(nb_item);
    SmallSpan<Int32> in_scan_data = scan_input_data.to1DSmallSpan();
    SmallSpan<Int32> out_scan_data = scan_output_data.to1DSmallSpan();
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_item)
      {
        auto [i] = iter();
        in_scan_data[i] = select_lambda(input_iter[i]) ? 1 : 0;
      };
    }
    queue->barrier();
    SyclScanner<false /*is_exclusive*/, Int32, ScannerSumOperator<Int32>> scanner;
    scanner.doScan(*queue, in_scan_data, out_scan_data, 0);
    // La valeur de 'out_data' pour le dernier élément (nb_item-1) contient la taille du filtre
    Int32 nb_output = out_scan_data[nb_item - 1];
    s.m_host_nb_out_storage[0] = nb_output;

    const bool do_verbose = false;
    if (do_verbose && nb_item < 1500)
      for (int i = 0; i < nb_item; ++i) {
        std::cout << "out_data i=" << i << " out_data=" << out_scan_data[i]
                  << " in_data=" << in_scan_data[i] << " value=" << input_iter[i] << "\n ";
      }
    // Copie depuis 'out_data' vers 'in_data' les indices correspondant au filtre
    // Comme 'output_iter' et 'input_iter' peuvent se chevaucher, il
    // faut faire une copie intermédiaire
    // TODO: détecter cela et ne faire la copie que si nécessaire.
    NumArray<DataType,MDDim1> out_copy(eMemoryRessource::Device);
    out_copy.resize(nb_output);
    auto out_copy_view = out_copy.to1DSpan();
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_item)
      {
        auto [i] = iter();
        if (in_scan_data[i] == 1)
          out_copy_view[out_scan_data[i] - 1] = input_iter[i];
      };
    }
    {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_output)
      {
        auto [i] = iter();
        output_iter[i] = out_copy_view[i];
      };
    }
    // Obligatoire à cause de 'out_copy'. On pourra le supprimer avec une
    // allocation temporaire.
    queue->barrier();
#endif
  }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour effectuer un filtrage
 *
 * \a DataType est le type de donnée.
 * \a FlagType est le type du tableau de filtre.
 */
template <typename DataType, typename FlagType, typename OutputDataType>
class GenericFilteringFlag
{
 public:

  void apply(GenericFilteringBase& s, SmallSpan<const DataType> input, SmallSpan<OutputDataType> output, SmallSpan<const FlagType> flag)
  {
    const Int32 nb_item = input.size();
    if (output.size() != nb_item)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_item, output.size());
    [[maybe_unused]] const DataType* input_data = input.data();
    [[maybe_unused]] DataType* output_data = output.data();
    [[maybe_unused]] const FlagType* flag_data = flag.data();
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue* queue = s.m_queue;
    ARCANE_CHECK_POINTER(queue);
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

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::Flagged(s.m_algo_storage.address(), temp_storage_size,
                                                     input_data, flag_data, output_data, nb_out_ptr, nb_item, stream));
      s._copyDeviceNbOutToHostNbOut();
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

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();

      ARCANE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_data, flag_data, output_data,
                                       nb_out_ptr, nb_item, stream));
      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCANE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      impl::IndexIterator iter2(0);
      auto filter_lambda = [=](Int32 input_index) -> bool { return flag[input_index] != 0; };
      auto setter_lambda = [=](Int32 input_index, Int32 output_index) { output[output_index] = input[input_index]; };
      GenericFilteringBase::SetterLambdaIterator<decltype(setter_lambda)> out(setter_lambda);
      SyclGenericFilteringImpl::apply(s, nb_item, iter2, out, filter_lambda);
    } break;
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

  template <typename SelectLambda, typename InputIterator, typename OutputIterator>
  void apply(GenericFilteringBase& s, Int32 nb_item, InputIterator input_iter, OutputIterator output_iter,
             const SelectLambda& select_lambda, const TraceInfo& trace_info)
  {
    eExecutionPolicy exec_policy = eExecutionPolicy::Sequential;
    RunQueue* queue = s.m_queue;
    ARCANE_CHECK_POINTER(queue);
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

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      ARCANE_CHECK_CUDA(::cub::DeviceSelect::If(s.m_algo_storage.address(), temp_storage_size,
                                                input_iter, output_iter, nb_out_ptr, nb_item,
                                                select_lambda, stream));
      s._copyDeviceNbOutToHostNbOut();
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

      s._allocateTemporaryStorage(temp_storage_size);
      nb_out_ptr = s._getDeviceNbOutPointer();
      ARCANE_CHECK_HIP(rocprim::select(s.m_algo_storage.address(), temp_storage_size, input_iter, output_iter,
                                       nb_out_ptr, nb_item, select_lambda, 0));
      s._copyDeviceNbOutToHostNbOut();
    } break;
#endif
#if defined(ARCANE_COMPILING_SYCL)
    case eExecutionPolicy::SYCL: {
      SyclGenericFilteringImpl::apply(s, nb_item, input_iter, output_iter, select_lambda);
    } break;
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
 */
class GenericFilterer
: private impl::GenericFilteringBase
{

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
  template <typename InputDataType, typename OutputDataType, typename FlagType>
  void apply(SmallSpan<const InputDataType> input, SmallSpan<OutputDataType> output, SmallSpan<const FlagType> flag)
  {
    const Int32 nb_value = input.size();
    if (output.size() != nb_value)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_value, output.size());
    if (flag.size() != nb_value)
      ARCANE_FATAL("Sizes are not equals: input={0} flag={1}", nb_value, flag.size());

    _setCalled();
    if (_checkEmpty(nb_value))
      return;
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringFlag<InputDataType, FlagType, OutputDataType> gf;
    gf.apply(*base_ptr, input, output, flag);
  }

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
    const Int32 nb_value = input.size();
    if (output.size() != nb_value)
      ARCANE_FATAL("Sizes are not equals: input={0} output={1}", nb_value, output.size());

    _setCalled();
    if (_checkEmpty(nb_value))
      return;
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf;
    gf.apply(*base_ptr, nb_value, input.data(), output.data(), select_lambda, trace_info);
  }

  /*!
   * \brief Applique un filtre.
   *
   * Cette méthode est identique à Filterer::applyIf(SmallSpan<const DataType> input,
   * SmallSpan<DataType> output, const SelectLambda& select_lambda) mais permet de spécifier un
   * itérateur \a input_iter pour l'entrée et \a output_iter pour la sortie.
   * Le nombre d'entité en entrée est donné par \a nb_value.
   */
  template <typename InputIterator, typename OutputIterator, typename SelectLambda>
  void applyIf(Int32 nb_value, InputIterator input_iter, OutputIterator output_iter,
               const SelectLambda& select_lambda, const TraceInfo& trace_info = TraceInfo())
  {
    _setCalled();
    if (_checkEmpty(nb_value))
      return;
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf;
    gf.apply(*base_ptr, nb_value, input_iter, output_iter, select_lambda, trace_info);
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
    if (_checkEmpty(nb_value))
      return;
    impl::GenericFilteringBase* base_ptr = this;
    impl::GenericFilteringIf gf;
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
  bool _checkEmpty(Int32 nb_value)
  {
    if (nb_value == 0) {
      m_host_nb_out_storage[0] = 0;
      return true;
    }
    return false;
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
