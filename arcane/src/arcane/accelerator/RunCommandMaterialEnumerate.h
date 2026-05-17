// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandMaterialEnumerate.h                               (C) 2000-2026 */
/*                                                                           */
/* Exécution d'une boucle sur une liste de constituants.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Concurrency.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/ConstituentItemIndexedSelectionView.h"

#include "arcane/accelerator/KernelLauncher.h"
#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

#include "arccore/common/HostKernelRemainingArgsHelper.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index d'une boucle accélérateur sur les matériaux ou milieux.
 *
 * Cette classe permet de récupérer un EnvItemLocalId (pour un milieu) ou
 * un MatItemLocalId (pour un matériau) ainsi que le CellLocalId de la maille
 * globale associée.
 */
template <typename ConstituentItemLocalIdType_>
class ConstituentAndGlobalCellIteratorValue
{
 public:

  using ConstituentItemLocalIdType = ConstituentItemLocalIdType_;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 public:

  //! Structure interne simple pour éviter l'usage d'un std::tuple pour l'opérateur()
  struct Data
  {
   public:

    constexpr ARCCORE_HOST_DEVICE Data(ConstituentItemLocalIdType mvi, CellLocalId cid)
    : m_mvi(mvi)
    , m_cid(cid)
    {}

   public:

    ConstituentItemLocalIdType m_mvi;
    CellLocalId m_cid;
  };

 public:

  constexpr ARCCORE_HOST_DEVICE ConstituentAndGlobalCellIteratorValue(ConstituentItemLocalIdType mvi, CellLocalId cid, Int32 index)
  : m_internal_data{ mvi, cid }
  , m_index(index)
  {
  }

  /*!
   * \brief Cet opérateur permet de renvoyer le couple [ConstituentItemLocalIdType, CellLocalId].
   *
   * L'utilisation classique est :
   *
   * \code
   * // Pour un milieu \a envcellsv
   * // evi est de type EnvItemLocalId
   * cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, iter, envcellsv) {
   *   auto [evi, cid] = iter();
   * }
   * // Pour un matériau \a matcellsv
   * // mvi est de type MatItemLocalId
   * cmd << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, matcellsv) {
   *   auto [mvi, cid] = iter();
   * }
   * \endcode
   */
  constexpr ARCCORE_HOST_DEVICE Data operator()()
  {
    return m_internal_data;
  }

  //! Accesseur sur la partie MatVarIndex
  constexpr ARCCORE_HOST_DEVICE ConstituentItemLocalIdType varIndex() const { return m_internal_data.m_mvi; };

  //! Accesseur sur la partie cell local id
  constexpr ARCCORE_HOST_DEVICE CellLocalId globalCellId() const { return m_internal_data.m_cid; }

  //! Index de l'itération courante
  constexpr ARCCORE_HOST_DEVICE Int32 index() const { return m_index; }

 private:

  Data m_internal_data;
  Int32 m_index = -1;
};

//! Type de la valeur de l'itérateur pour RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell,...)
using EnvAndGlobalCellIteratorValue = ConstituentAndGlobalCellIteratorValue<EnvItemLocalId>;

//! Type de la valeur de l'itérateur pour RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell,...)
using MatAndGlobalCellIteratorValue = ConstituentAndGlobalCellIteratorValue<MatItemLocalId>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour les commandes sur les constituants.
 */
template <typename ContainerType_>
class ConstituentRunCommandBase2
{
 public:

  using ContainerType = ContainerType_;
  using ThatClass = ConstituentRunCommandBase2<ContainerType_>;
  using CommandType = ThatClass;

 public:

  static CommandType create(RunCommand& run_command, const ContainerType& items)
  {
    return CommandType(run_command, items);
  }

 private:

  // Uniquement appelable depuis 'Container'
  explicit ConstituentRunCommandBase2(RunCommand& command, const ContainerType& items)
  : m_command(command)
  , m_items(items)
  {
  }

 public:

  RunCommand& m_command;
  ContainerType m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour une commande sur les AllEnvCell.
 */
class AllEnvCellRunCommandContainer
{
 public:

  using ThatClass = AllEnvCellRunCommandContainer;
  using AllEnvCellVectorView = Arcane::Materials::AllEnvCellVectorView;
  using ContainerCreateViewType = AllEnvCellVectorView;
  using IteratorValueType = Arcane::Materials::AllEnvCell;
  using CommandType = ConstituentRunCommandBase2<ThatClass>;

 public:

  explicit AllEnvCellRunCommandContainer(ContainerCreateViewType view)
  : m_view(view)
  {
  }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_view.size(); }

  //! Accesseur pour le i-ème élément de la liste
  ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
  {
    return m_view[i];
  }

 private:

  AllEnvCellVectorView m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des conteneurs pour les command sur les constituants
 * (sauf pour les AllEnvCell).
 */
class ConstituentCommandContainerBase
{
 protected:

  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 protected:

  explicit ConstituentCommandContainerBase(ComponentItemVectorView view)
  : m_items(view)
  {
    m_nb_item = m_items.nbItem();
    m_matvar_indexes = m_items._matvarIndexes();
    m_global_cells_local_id = m_items._internalLocalIds();
  }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_nb_item; }

 protected:

  ComponentItemVectorView m_items;
  SmallSpan<const MatVarIndex> m_matvar_indexes;
  SmallSpan<const Int32> m_global_cells_local_id;
  Int32 m_nb_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur contenant les informations nécessaires pour une commande
 * sur les ConstituentItem.
 */
template <typename ConstituentItemLocalIdType_, typename ContainerCreateViewType_>
class ConstituentRunCommandContainer
: public ConstituentCommandContainerBase
{
 public:

  using ThatClass = ConstituentRunCommandContainer;
  using IteratorValueType = ConstituentItemLocalIdType_;
  using CommandType = ConstituentRunCommandBase2<ThatClass>;
  using ContainerCreateViewType = ContainerCreateViewType_;

 public:

  explicit ConstituentRunCommandContainer(ContainerCreateViewType view)
  : Impl::ConstituentCommandContainerBase(view)
  {
  }

 public:

  //! Accesseur pour le i-ème élément de la liste
  constexpr ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
  {
    return { ComponentItemLocalId(m_matvar_indexes[i]) };
  }
};

using EnvCellRunCommandContainer = ConstituentRunCommandContainer<Arcane::Materials::EnvItemLocalId, Arcane::Materials::EnvCellVectorView>;
using MatCellRunCommandContainer = ConstituentRunCommandContainer<Arcane::Materials::MatItemLocalId, Arcane::Materials::MatCellVectorView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur contenant les informations nécessaires pour une commande
 * permettant d'accéder à la fois au ConstituentItem et à l'entité globale.
 */
template <typename ConstituentItemLocalIdType_, typename ContainerCreateViewType_>
class ConstituentAndGlobalCellRunCommandContainer
: public ConstituentCommandContainerBase
{
 public:

  using ThatClass = ConstituentAndGlobalCellRunCommandContainer;
  using IteratorValueType = Arcane::Materials::ConstituentAndGlobalCellIteratorValue<ConstituentItemLocalIdType_>;
  using CommandType = ConstituentRunCommandBase2<ThatClass>;
  using ContainerCreateViewType = ContainerCreateViewType_;

 public:

  explicit ConstituentAndGlobalCellRunCommandContainer(ContainerCreateViewType_ view)
  : ConstituentCommandContainerBase(view)
  {
  }

 public:

  //! Accesseur pour le i-ème élément de la liste
  constexpr ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
  {
    return { ComponentItemLocalId(m_matvar_indexes[i]), CellLocalId(m_global_cells_local_id[i]), i };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using EnvAndGlobalCellRunCommandContainer = ConstituentAndGlobalCellRunCommandContainer<Arcane::Materials::EnvItemLocalId, Arcane::Materials::EnvCellVectorView>;
using MatAndGlobalCellRunCommandContainer = ConstituentAndGlobalCellRunCommandContainer<Arcane::Materials::MatItemLocalId, Arcane::Materials::MatCellVectorView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour les commande sur une sélection indexée de
 * ConstituentItem.
 */
template <typename ConstituentItemLocalIdType_, typename ContainerCreateViewType_>
class ConstituentIndexedSelectionRunCommandContainer
{
 public:

  using ThatClass = ConstituentIndexedSelectionRunCommandContainer;
  using IteratorValueType = ConstituentItemLocalIdType_;
  using CommandType = ConstituentRunCommandBase2<ThatClass>;
  using ContainerCreateViewType = ContainerCreateViewType_;

 public:

  explicit ConstituentIndexedSelectionRunCommandContainer(ContainerCreateViewType view)
  : m_view(view)
  {
  }

 public:

  //! Accesseur pour le i-ème élément de la liste
  constexpr ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
  {
    return { ComponentItemLocalId(m_view[i]) };
  }

  ARCCORE_HOST_DEVICE Int32 size() { return m_view.size(); }

 private:

  ContainerCreateViewType m_view;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using EnvIndexedSelectionRunCommandContainer = ConstituentIndexedSelectionRunCommandContainer<Arcane::Materials::EnvItemLocalId, Arcane::Materials::EnvCellVectorSelectionView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA_OR_HIP)
/*
 * Surcharge de la fonction de lancement de kernel pour GPU pour les ComponentItemLocalId et CellLocalId
 */
template <typename ContainerType, typename Lambda, typename... RemainingArgs> __global__ void
doMatContainerGPULambda(ContainerType items, Lambda func, RemainingArgs... remaining_args)
{
  auto privatizer = Impl::privatize(func);
  auto& body = privatizer.privateCopy();
  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  Impl::CudaHipKernelRemainingArgsHelper::applyAtBegin(i, remaining_args...);
  if (i < items.size()) {
    body(items[i], remaining_args...);
  }
  Impl::CudaHipKernelRemainingArgsHelper::applyAtEnd(i, remaining_args...);
}

#endif // ARCANE_COMPILING_CUDA_OR_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

template <typename ContainerType, typename Lambda, typename... RemainingArgs>
class DoMatContainerSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, SmallSpan<std::byte> shm_view,
                  ContainerType items, Lambda func,
                  RemainingArgs... remaining_args) const
  {
    auto privatizer = Impl::privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    Impl::SyclKernelRemainingArgsHelper::applyAtBegin(x, shm_view, remaining_args...);
    if (i < items.size()) {
      body(items[i], remaining_args...);
    }
    Impl::SyclKernelRemainingArgsHelper::applyAtEnd(x, shm_view, remaining_args...);
  }

  void operator()(sycl::id<1> x, ContainerType items, Lambda func) const
  {
    auto privatizer = Impl::privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x);
    if (i < items.size()) {
      body(items[i]);
    }
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ContainerType, typename Lambda, typename... RemainingArgs>
void _doConstituentItemsLambda(Int32 base_index, Int32 size, ContainerType items,
                               const Lambda& func, RemainingArgs... remaining_args)
{
  auto privatizer = Impl::privatize(func);
  auto& body = privatizer.privateCopy();

  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtBegin(remaining_args...);
  Int32 last_value = base_index + size;
  for (Int32 i = base_index; i < last_value; ++i) {
    body(items[i], remaining_args...);
  }
  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyAtEnd(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ContainerType_, typename... RemainingArgs>
class GenericConstituentCommandArgs
{
 public:

  using ContainerType = ContainerType_;
  using IteratorValueType = ContainerType::IteratorValueType;

 public:

  explicit GenericConstituentCommandArgs(const ContainerType& container, const RemainingArgs&... remaining_args)
  : m_container(container)
  , m_remaining_args(remaining_args...)
  {}

 public:

  ContainerType m_container;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ContainerType_, typename... RemainingArgs>
class GenericConstituentCommand
{
 public:

  using ContainerType = ContainerType_;
  using ConstituentCommandType = ContainerType::CommandType;

 public:

  explicit GenericConstituentCommand(const ConstituentCommandType& command)
  : m_command(command)
  {}
  explicit GenericConstituentCommand(const ConstituentCommandType& command,
                                     const std::tuple<RemainingArgs...>& remaining_args)
  : m_command(command)
  , m_remaining_args(remaining_args)
  {}

 public:

  ConstituentCommandType m_command;
  std::tuple<RemainingArgs...> m_remaining_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'énumération \a func sur la liste d'entité \a items.
 *
 * Le conteneur peut être issu de:
 * - EnvAndGlobalCellRunCommand
 * - EnvCellRunCommand
 * - MatAndGlobalCellRunCommand
 * - MatCellRunCommand
 */
template <typename ContainerType, typename Lambda, typename... RemainingArgs> void
_applyConstituentCells(RunCommand& command, ContainerType items, const Lambda& func, const RemainingArgs&... remaining_args)
{
  using namespace Arcane::Materials;
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = items.size();
  if (vsize == 0)
    return;

  Impl::RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    ARCCORE_KERNEL_CUDA_FUNC((doMatContainerGPULambda<ContainerType, Lambda, RemainingArgs...>),
                             launch_info, func, items, remaining_args...);
    break;
  case eExecutionPolicy::HIP:
    ARCCORE_KERNEL_HIP_FUNC((doMatContainerGPULambda<ContainerType, Lambda, RemainingArgs...>),
                            launch_info, func, items, remaining_args...);
    break;
  case eExecutionPolicy::SYCL:
    ARCCORE_KERNEL_SYCL_FUNC((impl::DoMatContainerSYCLLambda<ContainerType, Lambda, RemainingArgs...>{}),
                             launch_info, func, items, remaining_args...);
    break;
  case eExecutionPolicy::Sequential:
    _doConstituentItemsLambda(0, vsize, items, func, remaining_args...);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelFor(0, vsize, launch_info.loopRunInfo(),
                      [&](Int32 begin, Int32 size) {
                        _doConstituentItemsLambda(begin, size, items, func, remaining_args...);
                      });
    break;
  default:
    ARCCORE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConstituentCommandType, typename... RemainingArgs, typename Lambda>
void operator<<(const GenericConstituentCommand<ConstituentCommandType, RemainingArgs...>& c, const Lambda& func)
{
  if constexpr (sizeof...(RemainingArgs) > 0) {
    std::apply([&](auto... vs) {
      Impl::_applyConstituentCells(c.m_command.m_command, c.m_command.m_items, func, vs...);
    },
               c.m_remaining_args);
  }
  else
    Impl::_applyConstituentCells(c.m_command.m_command, c.m_command.m_items, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConstituentItemType, typename ConstituentItemContainerType, typename... RemainingArgs> auto
makeExtendedConstituentItemEnumeratorLoop(ConstituentItemType x,
                                          const ConstituentItemContainerType& container,
                                          const RemainingArgs&... remaining_args)
{
  auto container_instance = arcaneCreateRunCommandMaterialContainer(x, container);
  using TraitsType = decltype(container_instance); //RunCommandConstituentItemEnumeratorTraitsT<ConstituentItemType>;
  return GenericConstituentCommandArgs<TraitsType, RemainingArgs...>(container_instance, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un milieu et la maille globale associée
inline Accelerator::Impl::EnvAndGlobalCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(Arcane::Materials::EnvAndGlobalCell, Arcane::Materials::IMeshEnvironment* env)
{
  return Accelerator::Impl::EnvAndGlobalCellRunCommandContainer{ env->envView() };
}
inline Accelerator::Impl::EnvAndGlobalCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(Arcane::Materials::EnvAndGlobalCell, Arcane::Materials::EnvCellVectorView view)
{
  return Accelerator::Impl::EnvAndGlobalCellRunCommandContainer{ view };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau et la maille globale associée
inline Accelerator::Impl::MatAndGlobalCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(MatAndGlobalCell, IMeshMaterial* mat)
{
  return Accelerator::Impl::MatAndGlobalCellRunCommandContainer{ mat->matView() };
}
inline Accelerator::Impl::MatAndGlobalCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(MatAndGlobalCell, MatCellVectorView mat)
{
  return Accelerator::Impl::MatAndGlobalCellRunCommandContainer{ mat };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur les AllEvnCell
inline Accelerator::Impl::AllEnvCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(AllEnvCell, AllEnvCellVectorView items)
{
  return Accelerator::Impl::AllEnvCellRunCommandContainer{ items };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un milieu.
inline Accelerator::Impl::EnvCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(EnvCell, IMeshEnvironment* env)
{
  return Accelerator::Impl::EnvCellRunCommandContainer(env->envView());
}
inline Accelerator::Impl::EnvCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(EnvCell, EnvCellVectorView view)
{
  return Accelerator::Impl::EnvCellRunCommandContainer(view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau
inline Accelerator::Impl::MatCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(MatCell, IMeshMaterial* mat)
{
  return Accelerator::Impl::MatCellRunCommandContainer(mat->matView());
}
inline Accelerator::Impl::MatCellRunCommandContainer
arcaneCreateRunCommandMaterialContainer(MatCell, MatCellVectorView view)
{
  return Accelerator::Impl::MatCellRunCommandContainer(view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Accelerator::Impl::EnvIndexedSelectionRunCommandContainer
arcaneCreateRunCommandMaterialContainer(EnvCell, EnvCellVectorSelectionView view)
{
  return Accelerator::Impl::EnvIndexedSelectionRunCommandContainer{ view };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const Impl::GenericConstituentCommandArgs<TraitsType, RemainingArgs...>& args)
{
  using ContainerType = typename Impl::GenericConstituentCommandArgs<TraitsType, RemainingArgs...>::ContainerType;
  using CommandType = typename ContainerType::CommandType;
  using GenericCommandType = Impl::GenericConstituentCommand<CommandType, RemainingArgs...>;
  return GenericCommandType(CommandType::create(command, args.m_container), args.m_remaining_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const Impl::MatAndGlobalCellRunCommandContainer& view)
{
  using CommandType = Impl::MatAndGlobalCellRunCommandContainer::CommandType;
  return Impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const Impl::EnvAndGlobalCellRunCommandContainer& view)
{
  using CommandType = Impl::EnvAndGlobalCellRunCommandContainer::CommandType;
  return Impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const Impl::EnvCellRunCommandContainer& view)
{
  using CommandType = Impl::EnvCellRunCommandContainer::CommandType;
  return Impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const Impl::MatCellRunCommandContainer& view)
{
  using CommandType = Impl::MatCellRunCommandContainer::CommandType;
  return Impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define A_RUNCOMMAND_MAT_ENUMERATE_BUILDER_HELPER(ConstituentItemNameType, env_or_mat_container, ...) \
  ::Arcane::Accelerator::Impl::makeExtendedConstituentItemEnumeratorLoop(ConstituentItemNameType{}, env_or_mat_container __VA_OPT__(, __VA_ARGS__))

/*!
 * \brief Macro pour itérer sur un matériau ou un milieu
 *
 * \param ConstituentItemNameType est le type de l'énumérateur.
 * \param iter_name est le nom de l'itérateur
 * \param env_or_mat_container est le conteneur sur lequel on itère.
 *
 * Les paramètres supplémentaires sont utilisés pour les réductions
 * (voir \ref arcanedoc_acceleratorapi_reduction)
 *
 * \a ConstituentItemNameType doit être une des valeurs suivantes:
 *
 * - EnvAndGlobalCell
 * - EnvCell
 * - MatAndGlobalCell
 * - MatCell
 * - AllEnvCell
 *
 * Voir \ref arcanedoc_acceleratorapi_materials pour plus d'informations.
 */
#define RUNCOMMAND_MAT_ENUMERATE(ConstituentItemNameType, iter_name, env_or_mat_container, ...) \
  A_FUNCINFO << A_RUNCOMMAND_MAT_ENUMERATE_BUILDER_HELPER(ConstituentItemNameType, env_or_mat_container __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(A_RUNCOMMAND_MAT_ENUMERATE_BUILDER_HELPER(ConstituentItemNameType, env_or_mat_container __VA_OPT__(, __VA_ARGS__)))::IteratorValueType iter_name \
                                        __VA_OPT__(ARCCORE_RUNCOMMAND_REMAINING_FOR_EACH(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
