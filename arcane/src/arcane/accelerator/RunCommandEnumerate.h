// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandEnumerate.h                                       (C) 2000-2026 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une liste d'entités.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/RunCommand.h"
#include "arcane/accelerator/KernelLauncher.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Concurrency.h"

#include "arccore/common/HostKernelRemainingArgsHelper.h"

#include <concepts>

#if defined(ARCCORE_EXPERIMENTAL_GRID_STRIDE)
#include "arccore/common/StridedLoopRanges.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour la boucle accélérateur sur les entités.
 */
template <typename TraitsType_>
class ItemLocalIdsLoopRanges
{
 public:

  using TraitsType = TraitsType_;
  using BuilderType = TraitsType::BuilderType;

 public:

  explicit ItemLocalIdsLoopRanges(SmallSpan<const Int32> ids) : m_ids(ids){}
  constexpr SmallSpan<const Int32> ids() const { return m_ids; }
  constexpr Int64 nbElement() const { return m_ids.size(); }

 public:

  SmallSpan<const Int32> m_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename LoopBoundsType, typename Lambda, typename... RemainingArgs> __global__ void
doIndirectGPULambda2(LoopBoundsType bounds, Lambda func, RemainingArgs... remaining_args)
{
  // TODO: a supprimer quand il n'y aura plus les anciennes réductions
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;

  CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);

  if constexpr (requires { bounds.nbStride(); }) {
    // Test expérimental pour utiliser un pas de la taille
    // de la grille. Le nombre de pas est donné par bounds.nbStride().
    using BuilderType = LoopBoundsType::LoopBoundType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    Int32 nb_grid_stride = bounds.nbStride();
    Int32 offset = blockDim.x * gridDim.x;
    Int64 nb_item = bounds.nbOriginalElement();
    SmallSpan<const Int32> ids = bounds.originalLoop().ids();
#pragma unroll 4
    for (Int32 k = 0; k < nb_grid_stride; ++k) {
      Int32 true_i = i + (offset * k);
      if (true_i < nb_item) {
        LocalIdType lid(ids[true_i]);
        body(BuilderType::create(true_i, lid), remaining_args...);
      }
    }
  }
  else {
    using BuilderType = LoopBoundsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;

    SmallSpan<const Int32> ids = bounds.ids();
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid), remaining_args...);
    }
  }

  CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCCORE_COMPILING_CUDA_OR_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

//! Boucle 1D avec indirection
template <typename TraitsType, typename Lambda, typename... RemainingArgs>
class DoIndirectSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shared_memory,
                  SmallSpan<const Int32> ids, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    using BuilderType = TraitsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    SyclKernelRemainingArgsHelper::applyAtBegin(x, shared_memory, remaining_args...);
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid), remaining_args...);
    }
    SyclKernelRemainingArgsHelper::applyAtEnd(x, shared_memory, remaining_args...);
  }
  void operator()(sycl::id<1> x, SmallSpan<const Int32> ids, Lambda func) const
  {
    using BuilderType = TraitsType::BuilderType;
    using LocalIdType = BuilderType::ValueType;
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x);
    if (i < ids.size()) {
      LocalIdType lid(ids[i]);
      body(BuilderType::create(i, lid));
    }
  }
};

#endif

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IteratorWithIndexBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour un itérateur permettant de conserver l'index
 * de l'itération.
 */
template <typename T>
class IteratorWithIndex
: public IteratorWithIndexBase
{
 public:

  constexpr ARCCORE_HOST_DEVICE IteratorWithIndex(Int32 i, T v)
  : m_index(i)
  , m_value(v)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 index() const { return m_index; }
  constexpr ARCCORE_HOST_DEVICE T value() const { return m_value; }

 private:

  Int32 m_index;
  T m_value;
};

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

template <typename T>
class IterBuilderWithIndex
{
 public:

  using ValueType = T;

 public:

  constexpr ARCCORE_HOST_DEVICE static IteratorWithIndex<T> create(Int32 index, T value)
  {
    return IteratorWithIndex<T>(index, value);
  }
};

template <typename T>
class IterBuilderNoIndex
{
 public:

  using ValueType = T;

 public:

  constexpr ARCCORE_HOST_DEVICE static T create(Int32, T value)
  {
    return value;
  }
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Concept pour contraintre les valeurs dans RUNCOMMAND_ENUMERATE.
 *
 * Le type  doit être un type d'entité (Cell, Node, ...) ou un
 * type de numéro local (CellLocalId, NodeLocalId, ...)
 */
template <typename T>
concept RunCommandEnumerateIteratorConcept = std::derived_from<T, Item>
 || std::derived_from<T, ItemLocalId>
 || std::derived_from<T, IteratorWithIndexBase>;

//! Template pour connaitre le type d'entité associé à T
template <typename T>
class RunCommandItemEnumeratorSubTraitsT
{
 public:

  using ItemType = T;
  using ValueType = typename ItemTraitsT<ItemType>::LocalIdType;
  using BuilderType = Arcane::impl::IterBuilderNoIndex<ValueType>;
};

//! Spécialisation pour ItemLocalIdT.
template <typename T>
class RunCommandItemEnumeratorSubTraitsT<ItemLocalIdT<T>>
{
 public:

  using ItemType = typename ItemLocalIdT<T>::ItemType;
  using ValueType = ItemLocalIdT<T>;
  using BuilderType = Arcane::impl::IterBuilderNoIndex<ValueType>;
};

//! Spécialisation pour IteratorWithIndex<T>
template <typename T>
requires std::derived_from<T, ItemLocalId> class RunCommandItemEnumeratorSubTraitsT<IteratorWithIndex<T>>
{
 public:

  using ItemType = typename T::ItemType;
  using ValueType = IteratorWithIndex<T>;
  using BuilderType = Arcane::impl::IterBuilderWithIndex<T>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour RunCommandEnumerate.
 *
 * Le conteneur peut être soit un ItemVectorView, soit un ItemGroup.
 *
 * Le but de ce conteneur est d'éviter de faire le padding SIMD pour un
 * ItemGroup s'il est utilisé sur accélérateur. Comme le padding est
 * fait sur le CPU, cela induirait des transferts mémoire lorsqu'on utilise
 * la mémoire unifiée (ce qui est le cas par défaut).
 */
template <typename ItemType>
class RunCommandItemContainer
{
 public:

  explicit RunCommandItemContainer(const ItemGroupT<ItemType>& group)
  : m_item_group(group)
  , m_unpadded_vector_view(group._unpaddedView())
  {}
  explicit RunCommandItemContainer(const ItemVectorViewT<ItemType>& item_vector_view)
  : m_item_vector_view(item_vector_view)
  , m_unpadded_vector_view(item_vector_view)
  {
  }

 public:

  Int32 size() const { return m_unpadded_vector_view.size(); }
  SmallSpan<const Int32> localIds() const { return m_unpadded_vector_view.localIds(); }
  ItemVectorView paddedView() const
  {
    if (!m_item_group.null())
      return m_item_group._paddedView();
    return m_item_vector_view;
  }

 private:

  ItemVectorViewT<ItemType> m_item_vector_view;
  ItemGroupT<ItemType> m_item_group;
  ItemVectorViewT<ItemType> m_unpadded_vector_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un énumérateur d'une commande sur les entités.
 *
 * Cette classe doit être spécialisée et définir un type \a ValueType
 * qui correspond au type de retour de 'operator*' de l'énumérateur.
 *
 * \a IteratorValueType_ doit être un type d'entité (Cell, Node, ...) ou un
 * type de numéro local (CellLocalId, NodeLocalId, ...)
 */
template <RunCommandEnumerateIteratorConcept IteratorValueType_>
class RunCommandItemEnumeratorTraitsT
{
 public:

  using SubTraitsType = RunCommandItemEnumeratorSubTraitsT<IteratorValueType_>;
  using ItemType = typename SubTraitsType::ItemType;
  using LocalIdType = typename ItemTraitsT<ItemType>::LocalIdType;
  using ValueType = typename SubTraitsType::ValueType;
  using ContainerType = RunCommandItemContainer<ItemType>;
  using BuilderType = typename SubTraitsType::BuilderType;

 public:

  explicit RunCommandItemEnumeratorTraitsT(const ItemGroupT<ItemType>& group)
  : m_item_container(group)
  {}
  explicit RunCommandItemEnumeratorTraitsT(const ItemVectorViewT<ItemType>& vector_view)
  : m_item_container(vector_view)
  {
  }

 public:

  RunCommandItemContainer<ItemType> m_item_container;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename ContainerType, typename Lambda, typename... RemainingArgs>
void _doItemsLambda(Int32 base_index, ContainerType sub_items, const Lambda& func, RemainingArgs... remaining_args)
{
  using ItemType = TraitsType::ItemType;
  using BuilderType = TraitsType::BuilderType;
  using LocalIdType = BuilderType::ValueType;
  auto privatizer = Impl::privatize(func);
  auto& body = privatizer.privateCopy();

  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
  ENUMERATE_NO_TRACE_ (ItemType, iitem, sub_items) {
    body(BuilderType::create(iitem.index() + base_index, LocalIdType(iitem.itemLocalId())), remaining_args...);
  }
  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 */
template <typename TraitsType, typename Lambda, typename... RemainingArgs> void
_applyItems(RunCommand& command, typename TraitsType::ContainerType items,
            const Lambda& func, const RemainingArgs&... remaining_args)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Integer vsize = items.size();
  if (vsize == 0)
    return;
  using ItemType = typename TraitsType::ItemType;
  using LoopBoundType = Impl::ItemLocalIdsLoopRanges<TraitsType>;
  [[maybe_unused]] SmallSpan<const Int32> ids = items.localIds();
  [[maybe_unused]] LoopBoundType bounds(ids);

#if defined(ARCCORE_EXPERIMENTAL_GRID_STRIDE) && defined(ARCCORE_COMPILING_CUDA_OR_HIP)
  using TrueLoopBoundType = Impl::StridedLoopRanges<LoopBoundType>;
  TrueLoopBoundType bounds2(command.nbStride(), bounds);
  Impl::RunCommandLaunchInfo launch_info(command, bounds2.strideValue());
#else
  using TrueLoopBoundType = LoopBoundType;
  [[maybe_unused]] const TrueLoopBoundType& bounds2 = bounds;
  Impl::RunCommandLaunchInfo launch_info(command, vsize);
#endif

  const eExecutionPolicy exec_policy = launch_info.executionPolicy();

  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    ARCCORE_KERNEL_CUDA_FUNC((Impl::doIndirectGPULambda2<TrueLoopBoundType, Lambda, RemainingArgs...>),
                             launch_info, func, bounds2, remaining_args...);
    break;
  case eExecutionPolicy::HIP:
    ARCCORE_KERNEL_HIP_FUNC((Impl::doIndirectGPULambda2<TrueLoopBoundType, Lambda, RemainingArgs...>),
                            launch_info, func, bounds2, remaining_args...);
    break;
  case eExecutionPolicy::SYCL:
    ARCCORE_KERNEL_SYCL_FUNC((Impl::DoIndirectSYCLLambda<TraitsType, Lambda, RemainingArgs...>{}),
                             launch_info, func, ids, remaining_args...);
    break;
  case eExecutionPolicy::Sequential:
    impl::_doItemsLambda<TraitsType>(0, items.paddedView(), func, remaining_args...);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelForeach(items.paddedView(), launch_info.loopRunInfo(),
                          [&](ItemVectorViewT<ItemType> sub_items, Int32 base_index) {
                            impl::_doItemsLambda<TraitsType>(base_index, sub_items, func, remaining_args...);
                          });
    break;
  default:
    ARCCORE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... RemainingArgs>
class ItemRunCommandArgs
{
 public:

  ItemRunCommandArgs(const TraitsType& traits, const RemainingArgs&... remaining_args)
  : m_traits(traits)
  , m_remaining_args(remaining_args...)
  {
  }

 public:

  TraitsType m_traits;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename Lambda> void
run(RunCommand& command, const TraitsType& traits, const Lambda& func)
{
  impl::_applyItems<TraitsType>(command, traits.m_item_container, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... RemainingArgs>
class ItemRunCommand
{
 public:

  ItemRunCommand(RunCommand& command, const TraitsType& traits)
  : m_command(command)
  , m_traits(traits)
  {
  }

  ItemRunCommand(RunCommand& command, const TraitsType& traits, const std::tuple<RemainingArgs...>& remaining_args)
  : m_command(command)
  , m_traits(traits)
  , m_remaining_args(remaining_args)
  {
  }

 public:

  RunCommand& m_command;
  TraitsType m_traits;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType> auto
operator<<(RunCommand& command, const impl::RunCommandItemEnumeratorTraitsT<ItemType>& traits)
{
  using TraitsType = impl::RunCommandItemEnumeratorTraitsT<ItemType>;
  return ItemRunCommand<TraitsType>(command, traits);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Cette méthode est conservée pour compatibilité avec l'existant.
// A rendre obsolète mi-2024
template <typename ItemType> auto
operator<<(RunCommand& command, const ItemVectorViewT<ItemType>& items)
{
  using TraitsType = impl::RunCommandItemEnumeratorTraitsT<ItemType>;
  return ItemRunCommand<TraitsType>(command, TraitsType(items));
}

// Cette méthode est conservée pour compatibilité avec l'existant.
// A rendre obsolète mi-2024
template <typename ItemType> auto
operator<<(RunCommand& command, const ItemGroupT<ItemType>& items)
{
  using TraitsType = impl::RunCommandItemEnumeratorTraitsT<ItemType>;
  return ItemRunCommand<TraitsType>(command, TraitsType(items));
}

template <typename TraitsType, typename Lambda>
void operator<<(ItemRunCommand<TraitsType>& nr, const Lambda& f)
{
  run(nr.m_command, nr.m_traits, f);
}

template <typename TraitsType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const impl::ItemRunCommandArgs<TraitsType, RemainingArgs...>& args)
{
  return ItemRunCommand<TraitsType, RemainingArgs...>(command, args.m_traits, args.m_remaining_args);
}

template <typename TraitsType, typename Lambda, typename... RemainingArgs>
void operator<<(ItemRunCommand<TraitsType, RemainingArgs...>&& nr, const Lambda& f)
{
  if constexpr (sizeof...(RemainingArgs) > 0) {
    std::apply([&](auto... vs) { impl::_applyItems<TraitsType>(nr.m_command, nr.m_traits.m_item_container, f, vs...); }, nr.m_remaining_args);
  }
  else
    run(nr.m_command, nr.m_traits, f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemTypeName, typename ItemContainerType, typename... RemainingArgs> auto
makeExtendedItemEnumeratorLoop(const ItemContainerType& container_type,
                               const RemainingArgs&... remaining_args)
{
  using TraitsType = RunCommandItemEnumeratorTraitsT<ItemTypeName>;
  return ItemRunCommandArgs<TraitsType, RemainingArgs...>(TraitsType(container_type), remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour itérer sur accélérateur sur un groupe d'entités.
 *
 * Conceptuellement, cela est équivalent à la boucle suivante:
 *
 * \code
 * for( ItemTypeName iter_name : item_group ){
 *    ...
 * }
 * \endcode
 *
 * \a ItemTypeName doit être un type de numéro local (CellLocalId, NodeLocalId).
 * L'utilisation du nom du type de l'entité (Cell, Node, ...) est possible mais est
 * obsolète et il est équivalent à utiliser le type du numéro local (par exemple
 * il faut remplacer \a Cell par \a CellLocalId).
 * \a iter_name est le nom de la variable contenant la valeur courante de l'itérateur.
 * \a item_group est le nom du ItemGroup ou ItemVectorView associé
 * Les arguments supplémentaires servent à spécifier les réductions éventuelles
 */
#define RUNCOMMAND_ENUMERATE(ItemTypeName, iter_name, item_group, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedItemEnumeratorLoop<ItemTypeName>(item_group __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(::Arcane::Accelerator::impl::RunCommandItemEnumeratorTraitsT<ItemTypeName>::ValueType iter_name \
                                        __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
