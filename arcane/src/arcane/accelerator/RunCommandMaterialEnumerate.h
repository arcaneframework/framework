// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandMaterialEnumerate.h                               (C) 2000-2024 */
/*                                                                           */
/* Helpers et macros pour exécuter une boucle sur une liste d'envcell        */
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

#include "arcane/accelerator/RunQueueInternal.h"
#include "arcane/accelerator/RunCommand.h"
#include "arcane/accelerator/RunCommandLaunchInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

class MatCommandContainerBase
{
 protected:

  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 protected:

  explicit MatCommandContainerBase(ComponentItemVectorView view)
  : m_items(view)
  {
    _init();
  }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_nb_item; }

 protected:

  ComponentItemVectorView m_items;
  SmallSpan<const MatVarIndex> m_matvar_indexes;
  SmallSpan<const Int32> m_global_cells_local_id;
  Int32 m_nb_item = 0;

 private:

  void _init()
  {
    m_nb_item = m_items.nbItem();
    m_matvar_indexes = m_items._matvarIndexes();
    m_global_cells_local_id = m_items._internalLocalIds();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques d'un énumérateur d'une commande sur les matériaux/milieux.
 *
 * Cette classe doit être spécialisée et définit un type \a EnumeratorType
 * qui correspond à l'énumérateur.
 */
template <typename MatItemType>
class RunCommandMatItemEnumeratorTraitsT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Commande pour itérer sur les AllEnvCell.
 */
class AllEnvCellRunCommand
{
 public:

  using AllEnvCellVectorView = Arcane::Materials::AllEnvCellVectorView;
  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using AllEnvCell = Arcane::Materials::AllEnvCell;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  {
   public:

    explicit Container(AllEnvCellVectorView view)
    : m_view(view)
    {
    }

   public:

    AllEnvCellRunCommand createCommand(RunCommand& run_command) const
    {
      return AllEnvCellRunCommand(run_command, *this);
    }

   public:

    ARCCORE_HOST_DEVICE Int32 size() const { return m_view.size(); }

    //! Accesseur pour le i-ème élément de la liste
    ARCCORE_HOST_DEVICE AllEnvCell operator[](Int32 i) const
    {
      return m_view[i];
    }

   private:

    AllEnvCellVectorView m_view;
  };

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
/*!
 * \brief Commande pour itérer sur les EnvCell.
 */
class EnvCellRunCommand
{
 public:

  using EnvCellVectorView = Arcane::Materials::EnvCellVectorView;
  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using IMeshEnvironment = Arcane::Materials::IMeshEnvironment;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::MatCommandContainerBase
  {
   public:

    explicit Container(IMeshEnvironment* env)
    : impl::MatCommandContainerBase(env->envView())
    {
    }
    explicit Container(EnvCellVectorView view)
    : impl::MatCommandContainerBase(view)
    {
    }

   public:

    EnvCellRunCommand createCommand(RunCommand& run_command) const
    {
      return EnvCellRunCommand(run_command, *this);
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE ComponentItemLocalId operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]) };
    }
  };

 private:

  // Uniquement appelable depuis 'Container'
  explicit EnvCellRunCommand(RunCommand& command, const Container& items)
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
/*!
 * \brief Commande pour itérer sur les MatCell.
 */
class MatCellRunCommand
{
 public:

  using MatCellVectorView = Arcane::Materials::MatCellVectorView;
  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using IMeshMaterial = Arcane::Materials::IMeshMaterial;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 public:

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::MatCommandContainerBase
  {
   public:

    explicit Container(IMeshMaterial* mat)
    : impl::MatCommandContainerBase(mat->matView())
    {
    }
    explicit Container(MatCellVectorView view)
    : impl::MatCommandContainerBase(view)
    {
    }

   public:

    MatCellRunCommand createCommand(RunCommand& run_command) const
    {
      return MatCellRunCommand(run_command, *this);
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE ComponentItemLocalId operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]) };
    }
  };

 private:

  // Uniquement appelable depuis 'Container'
  explicit MatCellRunCommand(RunCommand& command, const Container& items)
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
/*!
 * \brief Commande pour itérer sur les EnvCell et récupérer aussi l'information
 * sur la maille globale associée.
 */
class EnvAndGlobalCellRunCommand
{
 public:

  using EnvCellVectorView = Arcane::Materials::EnvCellVectorView;
  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using IMeshEnvironment = Arcane::Materials::IMeshEnvironment;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 public:

  /*!
   * \brief Classe helper pour l'accès au MatVarIndex et au CellLocalId à travers les
   *        RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell...
   */
  class Accessor
  {

   public:

    //! Struct interne simple pour éviter l'usage d'un std::tuple pour l'opérateur()
    struct Data
    {
      ComponentItemLocalId m_mvi;
      CellLocalId m_cid;
    };

   public:

    constexpr ARCCORE_HOST_DEVICE Accessor(ComponentItemLocalId mvi, CellLocalId cid, Int32 index)
    : m_internal_data{ mvi, cid }
    , m_index(index)
    {
    }

    /*!
     * \brief Cet opérateur permet de renvoyer le couple [MatVarIndex, LocalCellId].
     *
     * L'utilisation classique est :
     *
     * \code
     * cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv) {
     *   auto [mvi, cid] = evi();
     * }
     * \endcode
     *
     * où evi est le type de cette classe
     */
    ARCCORE_HOST_DEVICE Data operator()()
    {
      return { m_internal_data.m_mvi, m_internal_data.m_cid };
    }

    //! Accesseur sur la partie MatVarIndex
    ARCCORE_HOST_DEVICE ComponentItemLocalId varIndex() const { return m_internal_data.m_mvi; };

    //! Accesseur sur la partie cell local id
    ARCCORE_HOST_DEVICE CellLocalId globalCellId() const { return m_internal_data.m_cid; }

    //! Index de l'itération courante
    ARCCORE_HOST_DEVICE Int32 index() const { return m_index; }

   private:

    Data m_internal_data;
    Int32 m_index = -1;
  };

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::MatCommandContainerBase
  {
   public:

    explicit Container(IMeshEnvironment* env)
    : impl::MatCommandContainerBase(env->envView())
    {
    }
    explicit Container(EnvCellVectorView view)
    : impl::MatCommandContainerBase(view)
    {
    }

   public:

    EnvAndGlobalCellRunCommand createCommand(RunCommand& run_command) const
    {
      return EnvAndGlobalCellRunCommand(run_command, *this);
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE Accessor operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]), CellLocalId(m_global_cells_local_id[i]), i };
    }
  };

 private:

  // Uniquement appelable depuis 'Container'
  explicit EnvAndGlobalCellRunCommand(RunCommand& command, const Container& items)
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
/*!
 * \brief Commande pour itérer sur les MatCell et récupérer aussi l'information
 * sur la maille globale associée.
 */
class MatAndGlobalCellRunCommand
{
 public:

  using MatCellVectorView = Arcane::Materials::MatCellVectorView;
  using ComponentItemVectorView = Arcane::Materials::ComponentItemVectorView;
  using IMeshMaterial = Arcane::Materials::IMeshMaterial;
  using ComponentItemLocalId = Arcane::Materials::ComponentItemLocalId;
  using MatVarIndex = Arcane::Materials::MatVarIndex;

 public:

  /*!
   * \brief Classe helper pour l'accès au MatVarIndex et au CellLocalId à travers les
   *        RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell...
   */
  class Accessor
  {

   public:

    //! Struct interne simple pour éviter l'usage d'un std::tuple pour l'opérateur()
    struct Data
    {
      ComponentItemLocalId m_mvi;
      CellLocalId m_cid;
    };

   public:

    constexpr ARCCORE_HOST_DEVICE Accessor(ComponentItemLocalId mvi, CellLocalId cid, Int32 index)
    : m_internal_data{ mvi, cid }
    , m_index(index)
    {
    }

    /*!
     * \brief Cet opérateur permet de renvoyer le couple [MatVarIndex, LocalCellId].
     *
     * L'utilisation classique est :
     *
     * \code
     * cmd << RUNCOMMAND_MAT_ENUMERATE(EnvAndGlobalCell, evi, envcellsv) {
     *   auto [mvi, cid] = evi();
     * }
     * \endcode
     *
     * où evi est le type de cette classe
     */
    ARCCORE_HOST_DEVICE Data operator()()
    {
      return { m_internal_data.m_mvi, m_internal_data.m_cid };
    }

    ///! Accesseur sur la partie MatVarIndex
    ARCCORE_HOST_DEVICE ComponentItemLocalId varIndex() const { return m_internal_data.m_mvi; };

    ///! Accesseur sur la partie cell local id
    ARCCORE_HOST_DEVICE CellLocalId globalCellId() const { return m_internal_data.m_cid; }

    //! Index de l'itération courante
    ARCCORE_HOST_DEVICE Int32 index() const { return m_index; }

   private:

    Data m_internal_data;
    Int32 m_index = -1;
  };

  /*!
   * \brief Conteneur contenant les informations nécessaires pour la commande.
   */
  class Container
  : public impl::MatCommandContainerBase
  {
   public:

    explicit Container(IMeshMaterial* env)
    : impl::MatCommandContainerBase(env->matView())
    {
    }
    explicit Container(MatCellVectorView view)
    : impl::MatCommandContainerBase(view)
    {
    }

   public:

    MatAndGlobalCellRunCommand createCommand(RunCommand& run_command) const
    {
      return MatAndGlobalCellRunCommand(run_command, *this);
    }

   public:

    //! Accesseur pour le i-ème élément de la liste
    constexpr ARCCORE_HOST_DEVICE Accessor operator[](Int32 i) const
    {
      return { ComponentItemLocalId(m_matvar_indexes[i]), CellLocalId(m_global_cells_local_id[i]), i };
    }
  };

 private:

  // Uniquement appelable depuis 'Container'
  explicit MatAndGlobalCellRunCommand(RunCommand& command, const Container& items)
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

//! Spécialisation pour une vue sur un milieu et la maille globale associée
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::EnvAndGlobalCell>
{
 public:

  using EnumeratorType = EnvAndGlobalCellRunCommand::Accessor;
  using ContainerType = EnvAndGlobalCellRunCommand::Container;
  using MatCommandType = EnvAndGlobalCellRunCommand;

 public:

  static ContainerType createContainer(const Arcane::Materials::EnvCellVectorView& items)
  {
    return ContainerType{ items };
  }
  static ContainerType createContainer(Arcane::Materials::IMeshEnvironment* env)
  {
    return ContainerType{ env };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau et la maille globale associée
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::MatAndGlobalCell>
{
 public:

  using EnumeratorType = MatAndGlobalCellRunCommand::Accessor;
  using ContainerType = MatAndGlobalCellRunCommand::Container;
  using MatCommandType = MatAndGlobalCellRunCommand;

 public:

  static ContainerType createContainer(const Arcane::Materials::MatCellVectorView& items)
  {
    return ContainerType{ items };
  }
  static ContainerType createContainer(Arcane::Materials::IMeshMaterial* env)
  {
    return ContainerType{ env };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur les AllEvnCell
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::AllEnvCell>
{
 public:

  using EnumeratorType = Arcane::Materials::AllEnvCell;
  using ContainerType = AllEnvCellRunCommand::Container;
  using MatCommandType = AllEnvCellRunCommand;

 public:

  static ContainerType createContainer(const Arcane::Materials::AllEnvCellVectorView& items)
  {
    return ContainerType{ items };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un milieu.
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::EnvCell>
{
 public:

  using EnumeratorType = Arcane::Materials::ComponentItemLocalId;
  using ContainerType = EnvCellRunCommand::Container;
  using MatCommandType = EnvCellRunCommand;

 public:

  static ContainerType createContainer(const Arcane::Materials::EnvCellVectorView& items)
  {
    return ContainerType{ items };
  }
  static ContainerType createContainer(Arcane::Materials::IMeshEnvironment* env)
  {
    return ContainerType{ env };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Spécialisation pour une vue sur un matériau
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::MatCell>
{
 public:

  using EnumeratorType = Arcane::Materials::ComponentItemLocalId;
  using ContainerType = MatCellRunCommand::Container;
  using MatCommandType = MatCellRunCommand;

 public:

  static ContainerType createContainer(const Arcane::Materials::MatCellVectorView& items)
  {
    return ContainerType{ items };
  }
  static ContainerType createContainer(Arcane::Materials::IMeshMaterial* mat)
  {
    return ContainerType{ mat };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
/*
 * Surcharge de la fonction de lancement de kernel pour GPU pour les ComponentItemLocalId et CellLocalId
 */
template <typename ContainerType, typename Lambda, typename... ReducerArgs> __global__ void
doMatContainerGPULambda(ContainerType items, Lambda func, ReducerArgs... reducer_args)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < items.size()) {
    body(items[i], reducer_args...);
  }
  KernelReducerHelper::applyReducerArgs(i, reducer_args...);
}

#endif // ARCANE_COMPILING_CUDA || ARCANE_COMPILING_HIP

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

template <typename ContainerType, typename Lambda, typename... ReducerArgs>
class DoMatContainerSYCLLambda
{
 public:

  void operator()(sycl::nd_item<1> x, ContainerType items, Lambda func, ReducerArgs... reducer_args) const
  {
    auto privatizer = privatize(func);
    auto& body = privatizer.privateCopy();

    Int32 i = static_cast<Int32>(x.get_global_id(0));
    if (i < items.size()) {
      body(items[i], reducer_args...);
    }
    KernelReducerHelper::applyReducerArgs(x, reducer_args...);
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

template <typename ContainerType, typename Lambda, typename... ReducerArgs>
void _doMatItemsLambda(Int32 base_index, Int32 size, ContainerType items, const Lambda& func, ReducerArgs... reducer_args)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 last_value = base_index + size;
  for (Int32 i = base_index; i < last_value; ++i) {
    body(items[i], reducer_args...);
  }
  ::Arcane::impl::HostReducerHelper::applyReducerArgs(reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename TraitsType, typename... ReducerArgs>
class GenericMatCommandArgs
{
 public:

  using ContainerType = typename TraitsType::ContainerType;
  using MatCommandType = typename TraitsType::MatCommandType;

 public:

  explicit GenericMatCommandArgs(const ContainerType& container, const ReducerArgs&... reducer_args)
  : m_container(container)
  , m_reducer_args(reducer_args...)
  {}

 public:

  ContainerType m_container;
  std::tuple<ReducerArgs...> m_reducer_args;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename MatCommandType, typename... ReducerArgs>
class GenericMatCommand
{
 public:

  using ContainerType = typename MatCommandType::Container;

 public:

  explicit GenericMatCommand(const MatCommandType& mat_command)
  : m_mat_command(mat_command)
  {}
  explicit GenericMatCommand(const MatCommandType& mat_command, const std::tuple<ReducerArgs...>& reducer_args)
  : m_mat_command(mat_command)
  , m_reducer_args(reducer_args)
  {}

 public:

  MatCommandType m_mat_command;
  std::tuple<ReducerArgs...> m_reducer_args;
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
template <typename ContainerType, typename Lambda, typename... ReducerArgs> void
_applyEnvCells(RunCommand& command, ContainerType items, const Lambda& func, const ReducerArgs&... reducer_args)
{
  using namespace Arcane::Materials;
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = items.size();
  if (vsize == 0)
    return;

  RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.computeLoopRunInfo();
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(doMatContainerGPULambda) < ContainerType, Lambda, ReducerArgs... >, func, items, reducer_args...);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(doMatContainerGPULambda) < ContainerType, Lambda, ReducerArgs... >, func, items, reducer_args...);
    break;
  case eExecutionPolicy::SYCL:
    _applyKernelSYCL(launch_info, ARCANE_KERNEL_SYCL_FUNC(impl::DoMatContainerSYCLLambda) < ContainerType, Lambda, ReducerArgs... > {}, func, items, reducer_args...);
    break;
  case eExecutionPolicy::Sequential:
    _doMatItemsLambda(0, vsize, items, func, reducer_args...);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelFor(0, vsize, launch_info.loopRunInfo(),
                      [&](Int32 begin, Int32 size) {
                        _doMatItemsLambda(begin, size, items, func, reducer_args...);
                      });
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'", exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename MatCommandType, typename... ReducerArgs, typename Lambda>
void operator<<(const GenericMatCommand<MatCommandType, ReducerArgs...>& c, const Lambda& func)
{
  if constexpr (sizeof...(ReducerArgs) > 0) {
    std::apply([&](auto... vs) { impl::_applyEnvCells(c.m_mat_command.m_command, c.m_mat_command.m_items, func, vs...); }, c.m_reducer_args);
  }
  else
    impl::_applyEnvCells(c.m_mat_command.m_command, c.m_mat_command.m_items, func);
}

template <typename MatItemType, typename MatItemContainerType, typename... ReducerArgs> auto
makeExtendedMatItemEnumeratorLoop(const MatItemContainerType& container,
                                  const ReducerArgs&... reducer_args)
{
  using TraitsType = RunCommandMatItemEnumeratorTraitsT<MatItemType>;
  return GenericMatCommandArgs<TraitsType, ReducerArgs...>(TraitsType::createContainer(container), reducer_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

template <typename TraitsType, typename... ReducerArgs> auto
operator<<(RunCommand& command, const impl::GenericMatCommandArgs<TraitsType, ReducerArgs...>& args)
{
  using MatCommandType = typename TraitsType::MatCommandType;
  return impl::GenericMatCommand<MatCommandType, ReducerArgs...>(args.m_container.createCommand(command), args.m_reducer_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline auto
operator<<(RunCommand& command, const impl::MatAndGlobalCellRunCommand::Container& view)
{
  return impl::GenericMatCommand<impl::MatAndGlobalCellRunCommand>(view.createCommand(command));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline auto
operator<<(RunCommand& command, const impl::EnvAndGlobalCellRunCommand::Container& view)
{
  return impl::GenericMatCommand<impl::EnvAndGlobalCellRunCommand>(view.createCommand(command));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline auto
operator<<(RunCommand& command, const impl::EnvCellRunCommand::Container& view)
{
  return impl::GenericMatCommand<impl::EnvCellRunCommand>(view.createCommand(command));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline auto
operator<<(RunCommand& command, const impl::MatCellRunCommand::Container& view)
{
  return impl::GenericMatCommand<impl::MatCellRunCommand>(view.createCommand(command));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer sur un matériau ou un milieu
#define RUNCOMMAND_MAT_ENUMERATE(MatItemNameType, iter_name, env_or_mat_vector, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedMatItemEnumeratorLoop<MatItemNameType>(env_or_mat_vector __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(::Arcane::Accelerator::impl::RunCommandMatItemEnumeratorTraitsT<MatItemNameType>::EnumeratorType iter_name \
                                        __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
