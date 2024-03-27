// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandEnumerate.h                                       (C) 2000-2024 */
/*                                                                           */
/* Macros pour exécuter une boucle sur une liste d'entités.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunQueueInternal.h"

#include "arcane/utils/ArcaneCxx20.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Concurrency.h"

#include <concepts>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IteratorWithIndexBase
{
};

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
concept RunCommandEnumerateIteratorConcept = std::derived_from<T, Item> || std::derived_from<T, ItemLocalId> || std::derived_from<T, IteratorWithIndexBase>;

//! Template pour connaitre le type d'entité associé à T
template <typename T>
class RunCommandItemEnumeratorSubTraitsT
{
 public:

  using ItemType = T;
  using ValueType = ItemTraitsT<ItemType>::LocalIdType;
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
  using LocalIdType = ItemTraitsT<ItemType>::LocalIdType;
  using ValueType = typename SubTraitsType::ValueType;
  using ContainerType = ItemVectorViewT<ItemType>;
  using BuilderType = SubTraitsType::BuilderType;

 public:

  explicit RunCommandItemEnumeratorTraitsT(const ItemGroupT<ItemType>& group)
  : m_items(group.view())
  {}
  explicit RunCommandItemEnumeratorTraitsT(const ItemVectorViewT<ItemType>& vector_view)
  : m_items(vector_view)
  {
  }

 public:

  ItemVectorViewT<ItemType> m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename BuilderType, typename ContainerType, typename Lambda>
void _doIndirectThreadLambda(Int32 base_index, ContainerType sub_items, const Lambda& func)
{
  using LocalIdType = BuilderType::ValueType;
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  ENUMERATE_ITEM (iitem, sub_items) {
    body(BuilderType::create(iitem.index() + base_index, LocalIdType(iitem.itemLocalId())));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 */
template <typename TraitsType, typename Lambda> void
_applyItems(RunCommand& command, typename TraitsType::ContainerType items, const Lambda& func)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Integer vsize = items.size();
  if (vsize == 0)
    return;
  using LocalIdType = TraitsType::LocalIdType;
  using ItemType = TraitsType::ItemType;
  using BuilderType = TraitsType::BuilderType;
  impl::RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.computeLoopRunInfo(vsize);
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(doIndirectGPULambda) < BuilderType, Lambda >, func, items.localIds());
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(doIndirectGPULambda) < BuilderType, Lambda >, func, items.localIds());
    break;
  case eExecutionPolicy::Sequential:
    ENUMERATE_NO_TRACE_(ItemType, iitem, items)
    {
      func(BuilderType::create(iitem.index(), LocalIdType(iitem.itemLocalId())));
    }
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelForeach(items, launch_info.loopRunInfo(),
                          [&](ItemVectorViewT<ItemType> sub_items, Int32 base_index) {
                            impl::_doIndirectThreadLambda<BuilderType>(base_index, sub_items, func);
                          });
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

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
  impl::_applyItems<TraitsType>(command, traits.m_items, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType>
class ItemRunCommand
{
 public:

  ItemRunCommand(RunCommand& command, const TraitsType& traits)
  : m_command(command)
  , m_traits(traits)
  {
  }
  RunCommand& m_command;
  TraitsType m_traits;
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename Lambda>
void operator<<(ItemRunCommand<TraitsType>&& nr, const Lambda& f)
{
  run(nr.m_command, nr.m_traits, f);
}

template <typename TraitsType, typename Lambda>
void operator<<(ItemRunCommand<TraitsType>& nr, const Lambda& f)
{
  run(nr.m_command, nr.m_traits, f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

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
 */
#define RUNCOMMAND_ENUMERATE(ItemTypeName, iter_name, item_group) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::RunCommandItemEnumeratorTraitsT<ItemTypeName>(item_group) \
             << [=] ARCCORE_HOST_DEVICE(::Arcane::Accelerator::impl::RunCommandItemEnumeratorTraitsT<ItemTypeName>::ValueType iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
