// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandMaterialEnumerate.h                               (C) 2000-2023 */
/*                                                                           */
/* Helpers et macros pour exécuter une boucle sur une liste d'envcell        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
#define ARCANE_ACCELERATOR_RUNCOMMANDMATERIALENUMERATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Concurrency.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/core/materials/MatItem.h"
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

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

    constexpr ARCCORE_HOST_DEVICE Accessor(ComponentItemLocalId mvi, CellLocalId cid)
    : m_internal_data{ mvi, cid }
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
    ARCCORE_HOST_DEVICE ComponentItemLocalId varIndex() { return m_internal_data.m_mvi; };

    ///! Accesseur sur la partie cell local id
    ARCCORE_HOST_DEVICE CellLocalId globalCellId() { return m_internal_data.m_cid; }

   private:

    Data m_internal_data;
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
      return { ComponentItemLocalId(m_matvar_indexes[i]), CellLocalId(m_global_cells_local_id[i]) };
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

//! Spécialisation pour une vue sur un milieu et la maille globale associée
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::EnvAndGlobalCell>
{
 public:

  using EnumeratorType = EnvAndGlobalCellRunCommand::Accessor;
  using ContainerType = EnvAndGlobalCellRunCommand::Container;

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

//! Spécialisation pour une vue sur un milieu.
template <>
class RunCommandMatItemEnumeratorTraitsT<Arcane::Materials::EnvCell>
{
 public:

  using EnumeratorType = Arcane::Materials::ComponentItemLocalId;
  using ContainerType = EnvCellRunCommand::Container;

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

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
/*
 * Surcharge de la fonction de lancement de kernel pour GPU pour les ComponentItemLocalId et CellLocalId
 */
template <typename ContainerType, typename Lambda> __global__ void
doMatContainerGPULambda(ContainerType items, Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < items.size()) {
    body(items[i]);
  }
}

#endif // ARCANE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 *
 * Le container peut être issu de EnvAndGlobalCellRunCommand ou 'EnvCellRunCommand.
 */
template <typename ContainerType, typename Lambda> void
_applyEnvCells(RunCommand& command, ContainerType items, const Lambda& func)
{
  using namespace Arcane::Materials;
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = items.size();
  if (vsize == 0)
    return;

  RunCommandLaunchInfo launch_info(command, vsize);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  launch_info.computeLoopRunInfo(vsize);
  launch_info.beginExecute();
  switch (exec_policy) {
  case eExecutionPolicy::CUDA:
    _applyKernelCUDA(launch_info, ARCANE_KERNEL_CUDA_FUNC(doMatContainerGPULambda) < ContainerType, Lambda >, func, items);
    break;
  case eExecutionPolicy::HIP:
    _applyKernelHIP(launch_info, ARCANE_KERNEL_HIP_FUNC(doMatContainerGPULambda) < ContainerType, Lambda >, func, items);
    break;
  case eExecutionPolicy::Sequential:
    for (Int32 i = 0, n = vsize; i < n; ++i)
      func(items[i]);
    break;
  case eExecutionPolicy::Thread:
    arcaneParallelFor(0, vsize, launch_info.loopRunInfo(),
                      [&](Int32 begin, Int32 size) {
                        for (Int32 i = begin, n = (begin + size); i < n; ++i)
                          func(items[i]);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline EnvAndGlobalCellRunCommand
operator<<(RunCommand& command, const EnvAndGlobalCellRunCommand::Container& view)
{
  return view.createCommand(command);
}

template <typename Lambda>
void operator<<(EnvAndGlobalCellRunCommand&& nr, const Lambda& func)
{
  impl::_applyEnvCells(nr.m_command, nr.m_items, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline EnvCellRunCommand
operator<<(RunCommand& command, const EnvCellRunCommand::Container& view)
{
  return view.createCommand(command);
}

template <typename Lambda>
void operator<<(EnvCellRunCommand&& nr, const Lambda& func)
{
  impl::_applyEnvCells(nr.m_command, nr.m_items, func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour itérer un matériau ou un milieu
#define RUNCOMMAND_MAT_ENUMERATE(MatItemNameType, iter_name, env_or_mat_vector) \
  A_FUNCINFO << Arcane::Accelerator::RunCommandMatItemEnumeratorTraitsT<MatItemNameType>::createContainer(env_or_mat_vector) \
             << [=] ARCCORE_HOST_DEVICE(Arcane::Accelerator::RunCommandMatItemEnumeratorTraitsT<MatItemNameType>::EnumeratorType iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
