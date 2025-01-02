// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandMaterialEnumerate.h                               (C) 2000-2024 */
/*                                                                           */
/* Exécution d'une boucle sur une liste de constituants.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

#include "arcane/core/Concurrency.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/MatItemEnumerator.h"

#include "arcane/accelerator/KernelLauncher.h"
#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

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

  //! Struct interne simple pour éviter l'usage d'un std::tuple pour l'opérateur()
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

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Commande pour itérer sur les AllEnvCell.
 */
class AllEnvCellRunCommand
{
  using AllEnvCellVectorView = Arcane::Materials::AllEnvCellVectorView;

 public:

  using IteratorValueType = Arcane::Materials::AllEnvCell;
  using ContainerCreateViewType = AllEnvCellVectorView;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  {
   public:

    explicit Container(ContainerCreateViewType view)
    : m_view(view)
    {
    }

   public:
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

 public:

  static AllEnvCellRunCommand create(RunCommand& run_command, const Container& items)
  {
    return AllEnvCellRunCommand(run_command, items);
  }

 private:

  // Uniquement appelable depuis 'Container'
  explicit AllEnvCellRunCommand(RunCommand& command, const Container& items)
  : m_command(command)
  , m_items(items)
  {
  }

 public:

  RunCommand& m_command;
  Container m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
 * \brief Commande pour itérer sur les EnvCell ou MatCell.
 */
template <typename ConstituentItemLocalIdType_, typename ContainerCreateViewType_>
class ConstituentRunCommandBase
{
 public:

  using ThatClass = ConstituentRunCommandBase<ConstituentItemLocalIdType_, ContainerCreateViewType_>;
  using CommandType = ThatClass;
  using IteratorValueType = ConstituentItemLocalIdType_;
  using ContainerCreateViewType = ContainerCreateViewType_;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::ConstituentCommandContainerBase
  {
   public:

    explicit Container(ContainerCreateViewType view)
    : impl::ConstituentCommandContainerBase(view)
    {
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]) };
    }
  };

 public:

  static CommandType create(RunCommand& run_command, const Container& items)
  {
    return CommandType(run_command, items);
  }

 private:

  // Uniquement appelable depuis 'Container'
  explicit ConstituentRunCommandBase(RunCommand& command, const Container& items)
  : m_command(command)
  , m_items(items)
  {
  }

 public:

  RunCommand& m_command;
  Container m_items;
};

using EnvCellRunCommand = ConstituentRunCommandBase<Arcane::Materials::EnvItemLocalId, Arcane::Materials::EnvCellVectorView>;
using MatCellRunCommand = ConstituentRunCommandBase<Arcane::Materials::MatItemLocalId, Arcane::Materials::MatCellVectorView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour les commandes MatAndGlobalCell et EnvAndGlobalCell.
 */
template <typename ConstituentItemLocalIdType_, typename ContainerCreateViewType_>
class ConstituentAndGlobalCellRunCommandBase
{
 public:

  using ThatClass = ConstituentAndGlobalCellRunCommandBase<ConstituentItemLocalIdType_, ContainerCreateViewType_>;
  using CommandType = ThatClass;
  using IteratorValueType = Arcane::Materials::ConstituentAndGlobalCellIteratorValue<ConstituentItemLocalIdType_>;
  using ContainerCreateViewType = ContainerCreateViewType_;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::ConstituentCommandContainerBase
  {
   public:

    explicit Container(ContainerCreateViewType view)
    : impl::ConstituentCommandContainerBase(view)
    {
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE IteratorValueType operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]), CellLocalId(m_global_cells_local_id[i]), i };
    }
  };

 public:

  static CommandType create(RunCommand& run_command, const Container& items)
  {
    return CommandType(run_command, items);
  }

 private:

  // Uniquement appelable depuis 'Container'
  explicit ConstituentAndGlobalCellRunCommandBase(RunCommand& command, const Container& items)
  : m_command(command)
  , m_items(items)
  {
  }

 public:

  RunCommand& m_command;
  Container m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using EnvAndGlobalCellRunCommand = ConstituentAndGlobalCellRunCommandBase<Arcane::Materials::EnvItemLocalId, Arcane::Materials::EnvCellVectorView>;
using MatAndGlobalCellRunCommand = ConstituentAndGlobalCellRunCommandBase<Arcane::Materials::MatItemLocalId, Arcane::Materials::MatCellVectorView>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un énumérateur d'une commande sur les matériaux/milieux.
 *
 * Cette classe doit être spécialisée et définir les types suivants:
 * - CommandType
 * - IteratorValueType
 * - ContainerType
 * - ContainerCreateViewType
 */
template <typename MatItemType>
class RunCommandConstituentItemEnumeratorTraitsT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des caractéristiques des commandes sur les constituants.
 */
template <typename CommandType_>
class RunCommandConstituentItemTraitsBaseT
{
 public:

  using CommandType = CommandType_;
  using IteratorValueType = CommandType::IteratorValueType;
  using ContainerType = CommandType::Container;
  using ContainerCreateViewType = CommandType::ContainerCreateViewType;

 public:

  static ContainerType createContainer(const ContainerCreateViewType& items)
  {
    return ContainerType{ items };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un milieu et la maille globale associée
template <>
class RunCommandConstituentItemEnumeratorTraitsT<Arcane::Materials::EnvAndGlobalCell>
: public RunCommandConstituentItemTraitsBaseT<EnvAndGlobalCellRunCommand>
{
 public:

  using BaseClass = RunCommandConstituentItemTraitsBaseT<EnvAndGlobalCellRunCommand>;
  using BaseClass::createContainer;

  static ContainerType createContainer(Arcane::Materials::IMeshEnvironment* env)
  {
    return ContainerType{ env->envView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau et la maille globale associée
template <>
class RunCommandConstituentItemEnumeratorTraitsT<Arcane::Materials::MatAndGlobalCell>
: public RunCommandConstituentItemTraitsBaseT<MatAndGlobalCellRunCommand>
{
 public:

  using BaseClass = RunCommandConstituentItemTraitsBaseT<MatAndGlobalCellRunCommand>;
  using BaseClass::createContainer;

  static ContainerType createContainer(Arcane::Materials::IMeshMaterial* mat)
  {
    return ContainerType{ mat->matView() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur les AllEvnCell
template <>
class RunCommandConstituentItemEnumeratorTraitsT<Arcane::Materials::AllEnvCell>
: public RunCommandConstituentItemTraitsBaseT<AllEnvCellRunCommand>
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un milieu.
template <>
class RunCommandConstituentItemEnumeratorTraitsT<Arcane::Materials::EnvCell>
: public RunCommandConstituentItemTraitsBaseT<EnvCellRunCommand>
{
 public:

  using BaseClass = RunCommandConstituentItemTraitsBaseT<EnvCellRunCommand>;
  using BaseClass::createContainer;

 public:

  static ContainerType createContainer(Arcane::Materials::IMeshEnvironment* env)
  {
    return ContainerType(env->envView());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau
template <>
class RunCommandConstituentItemEnumeratorTraitsT<Arcane::Materials::MatCell>
: public RunCommandConstituentItemTraitsBaseT<MatCellRunCommand>
{
 public:

  using BaseClass = RunCommandConstituentItemTraitsBaseT<MatCellRunCommand>;
  using BaseClass::createContainer;

  static ContainerType createContainer(Arcane::Materials::IMeshMaterial* mat)
  {
    return ContainerType(mat->matView());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
/*
 * Surcharge de la fonction de lancement de kernel pour GPU pour les ComponentItemLocalId et CellLocalId
 */
template <typename ContainerType, typename Lambda, typename... RemainingArgs> __global__ void
doMatContainerGPULambda(ContainerType items, Lambda func, RemainingArgs... remaining_args)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < items.size()) {
    body(items[i], remaining_args...);
  }
  KernelRemainingArgsHelper::applyRemainingArgs(i, remaining_args...);
}

#endif // ARCANE_COMPILING_CUDA || ARCANE_COMPILING_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

template <typename ContainerType, typename Lambda, typename... RemainingArgs>
class DoMatContainerSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, ContainerType items, Lambda func, RemainingArgs... remaining_args) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    if (i < items.size()) {
      body(items[i], remaining_args...);
    }
    KernelRemainingArgsHelper::applyRemainingArgs(x, remaining_args...);
  }

  void operator()(sycl::id<1> x, ContainerType items, Lambda func) const
  {
    auto privatizer = privatize(func);
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
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 last_value = base_index + size;
  for (Int32 i = base_index; i < last_value; ++i) {
    body(items[i], remaining_args...);
  }
  ::Arcane::impl::HostReducerHelper::applyReducerArgs(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... RemainingArgs>
class GenericConstituentCommandArgs
{
 public:

  using ContainerType = typename TraitsType::ContainerType;

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

template <typename ConstituentCommandType, typename... RemainingArgs>
class GenericConstituentCommand
{
 public:

  using ContainerType = typename ConstituentCommandType::Container;

 public:

  explicit GenericConstituentCommand(const ConstituentCommandType& command)
  : m_command(command)
  {}
  explicit GenericConstituentCommand(const ConstituentCommandType& command, const std::tuple<RemainingArgs...>& remaining_args)
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

  RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(doMatContainerGPULambda) < ContainerType, Lambda, RemainingArgs... >,
                     func, items, remaining_args...);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(doMatContainerGPULambda) < ContainerType, Lambda, RemainingArgs... >,
                    func, items, remaining_args...);
    break;
  case eExecutionPolicy::SYCL:
    _applyKernelSYCL(launch_info, ARCANE_KERNEL_SYCL_FUNC(impl::DoMatContainerSYCLLambda) < ContainerType, Lambda, RemainingArgs... > {},
                     func, items, remaining_args...);
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
    ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
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
      impl::_applyConstituentCells(c.m_command.m_command, c.m_command.m_items, func, vs...);
    },
               c.m_remaining_args);
  }
  else
    impl::_applyConstituentCells(c.m_command.m_command, c.m_command.m_items, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ConstituentItemType, typename ConstituentItemContainerType, typename... RemainingArgs> auto
makeExtendedConstituentItemEnumeratorLoop(const ConstituentItemContainerType& container,
                                          const RemainingArgs&... remaining_args)
{
  using TraitsType = RunCommandConstituentItemEnumeratorTraitsT<ConstituentItemType>;
  return GenericConstituentCommandArgs<TraitsType, RemainingArgs...>(TraitsType::createContainer(container), remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... RemainingArgs> auto
operator<<(RunCommand& command, const impl::GenericConstituentCommandArgs<TraitsType, RemainingArgs...>& args)
{
  using CommandType = typename TraitsType::CommandType;
  using GenericCommandType = impl::GenericConstituentCommand<CommandType, RemainingArgs...>;
  return GenericCommandType(CommandType::create(command, args.m_container), args.m_remaining_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const impl::MatAndGlobalCellRunCommand::Container& view)
{
  using CommandType = impl::MatAndGlobalCellRunCommand;
  return impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const impl::EnvAndGlobalCellRunCommand::Container& view)
{
  using CommandType = impl::EnvAndGlobalCellRunCommand;
  return impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const impl::EnvCellRunCommand::Container& view)
{
  using CommandType = impl::EnvCellRunCommand;
  return impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: rendre obsolète (il faut utiliser la version générique)
inline auto
operator<<(RunCommand& command, const impl::MatCellRunCommand::Container& view)
{
  using CommandType = impl::MatCellRunCommand;
  return impl::GenericConstituentCommand<CommandType>(CommandType::create(command, view));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedConstituentItemEnumeratorLoop<ConstituentItemNameType>(env_or_mat_container __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(::Arcane::Accelerator::impl::RunCommandConstituentItemEnumeratorTraitsT<ConstituentItemNameType>::IteratorValueType iter_name \
                                        __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
