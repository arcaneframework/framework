// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.h                                        (C) 2000-2025 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_WORKGROUPLOOPRANGE_H
#define ARCANE_ACCELERATOR_WORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include "arccore/common/SequentialFor.h"

#if defined(ARCANE_COMPILING_CUDA)
#include <cooperative_groups.h>
#endif
#if defined(ARCANE_COMPILING_HIP)
#include <hip/hip_cooperative_groups.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class WorkGroupLoopRange;
class WorkGroupLoopContext;
class HostWorkItemBlock;
class SyclDeviceWorkItemBlock;
class DeviceWorkItemBlock;
class SyclWorkGroupLoopContext;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Représente un WorkItem dans le parallélisme hiérarchique.
 */
class WorkItem
{
  friend WorkGroupLoopContext;
  friend SyclDeviceWorkItemBlock;
  friend DeviceWorkItemBlock;
  friend HostWorkItemBlock;

 private:

  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE WorkItem(Int32 loop_index)
  : m_loop_index(loop_index)
  {}

 public:

  //! Index linéaire entre 0 et WorkGroupLoopRange::nbElement()
  constexpr Int32 linearIndex() const { return m_loop_index; }

 private:

  Int32 m_loop_index = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de WorkItem dans un WorkGroupLoopRange pour l'hôte.
 *
 * Contraitement à l'exécution sur accélérateur ou un seul WorkItem est
 * actif, l'hôte doit gérer un ensemble de WorkItem.
 *
 * Pour l'hôte, un bloc de WorkItem correspond toujours à l'ensemble
 * des WorkItem d'un groupe du WorkGroupLoopRange associé. Cela signifie
 * que nbActiveItem()==WorkGroupLoopRange::groupSize() (sauf pour le dernier
 * élément de l'itération si le nombre total d'élément n'est pas un multiple
 * de la taille d'un groupe).
 */
class HostWorkItemBlock
{
  friend WorkGroupLoopContext;
  friend SyclDeviceWorkItemBlock;
  friend DeviceWorkItemBlock;

 private:

  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE HostWorkItemBlock(Int32 loop_index, Int32 group_index, Int32 group_size)
  : m_loop_index(loop_index)
  , m_group_size(group_size)
  , m_group_index(group_index)
  {}

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  constexpr Int32 groupRank() const { return m_group_index; }

  //! Nombre de WorkItem dans un WorkGroup.
  constexpr Int32 groupSize() const { return m_group_size; }

  //! Rang du WorkItem actif dans son WorkGroup.
  constexpr Int32 activeWorkItemRankInGroup() const { return 0; }

  static constexpr bool isDevice() { return false; }

  void sync() {}

  constexpr Int32 nbActiveItem() const { return m_group_size; }
  WorkItem activeItem(Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_group_size);
    return WorkItem(m_loop_index);
  }

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de WorkItem dans un WorkGroupLoopRange pour un device CUDA ou ROCM.
 */
class DeviceWorkItemBlock
{
  friend WorkGroupLoopContext;

 private:

  /*!
   * \brief Constructeur pour le device (la taille du groupe est dans blockIdx.x).
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_thread_block()
   */
  explicit __device__ DeviceWorkItemBlock()
  : m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  /*!
   * \brief Rang du groupe du WorkItem dans la liste des WorkGroup.
   */
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  /*!
   * \brief Nombre de WorkItem dans un WorkGroup.
   */
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  //! Rang du WorkItem actif dans son WorkGroup.
  __device__ Int32 activeWorkItemRankInGroup() const { return m_thread_block.thread_index().x; }

  __device__ void sync() { m_thread_block.sync(); }

  constexpr __device__ bool isDevice() const { return true; }

  constexpr __device__ Int32 nbActiveItem() const { return 1; }
  __device__ WorkItem activeItem(Int32 index)
  {
    // Seulement valide pour index==0
    ARCANE_CHECK_AT(index, 1);
    return WorkItem(blockDim.x * blockIdx.x + threadIdx.x);
  }

 private:

  cooperative_groups::thread_block m_thread_block;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution d'une commande sur un ensemble de blocs.
 *
 * Cette classe est utilisée pour l'hôte et pour CUDA et ROCM/HIP.
 * La méthode block() est différente sur accélérateur et sur l'hôte ce qui
 * permet de particulariser le traitement de la commande.
 */
class WorkGroupLoopContext
{
  friend WorkGroupLoopRange;
  template <typename Lambda, typename... RemainingArgs>
  friend void arcaneSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, RemainingArgs... remaining_args);

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  explicit constexpr WorkGroupLoopContext(Int32 loop_index, Int32 group_index, Int32 group_size)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  {}

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContext() {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
  __device__ DeviceWorkItemBlock block() const { return DeviceWorkItemBlock(); }
#else
  HostWorkItemBlock block() const { return HostWorkItemBlock(m_loop_index, m_group_index, m_group_size); }
#endif

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Implémentation pour SYCL.
 *
 * L'équivalent de \a cooperative_groups::thread_group() avec SYCL
 * est le \a sycl::nd_item<1>. Il est plus compliqué à utiliser pour deux
 * raisons:
 *
 * - il n'y a pas dans SYCL un équivalent de
 * \a cooperative_groups::this_thread_block(). Il faut utiliser la valeur
 * de \a sycl::nb_item<1> passé en argument du noyau de calcul.
 * - il n'y a pas de constructeurs par défaut pour \a sycl::nb_item<1>.
 *
 * Pour contourner ces deux problèmes, on utilise un type spécifique pour
 * gérer les noyaux en SYCL. Heureusement, il est possible d'utiliser les
 * lambda template avec SYCL. On utilise donc deux types pour gérer
 * les noyaux selon qu'on s'exécute sur le device SYCL ou sur l'hôte.
 *
 * TODO: regarder si avec la macro SYCL_DEVICE_ONLY il n'est pas possible
 * d'avoir le même type comportant des champs différents
 */
#if defined(ARCANE_COMPILING_SYCL)

/*!
 * \brief Gère un bloc de WorkItem dans un WorkGroupLoopRange pour un device Sycl.
 */
class SyclDeviceWorkItemBlock
{
  friend SyclWorkGroupLoopContext;

 private:

  explicit SyclDeviceWorkItemBlock(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  Int32 groupRank() const { return static_cast<Int32>(m_nd_item.get_group(0)); }

  //! Nombre de WorkItem dans un WorkGroup.
  Int32 groupSize() { return static_cast<Int32>(m_nd_item.get_local_range(0)); }

  //! Rang du WorkItem actif dans le WorkGroup.
  Int32 activeWorkItemRankInGroup() const { return static_cast<Int32>(m_nd_item.get_local_id(0)); }

  void sync() { m_nd_item.barrier(); }

  constexpr bool isDevice() const { return true; }

  constexpr Int32 nbActiveItem() const { return 1; }
  WorkItem activeItem(Int32 index)
  {
    // Seulement valide pour index==0
    ARCANE_CHECK_AT(index, 1);
    return WorkItem(static_cast<Int32>(m_nd_item.get_group(0) * m_nd_item.get_local_range(0) + m_nd_item.get_local_id(0)));
  }

 private:

  sycl::nd_item<1> m_nd_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution d'une commande sur un ensemble de blocs.
 *
 * Cette classe est utilisée pour l'implémentation Sycl.
 */
class SyclWorkGroupLoopContext
{
  friend WorkGroupLoopRange;

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclWorkGroupLoopContext(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  SyclDeviceWorkItemBlock block() const { return SyclDeviceWorkItemBlock(m_nd_item); }

 private:

  sycl::nd_item<1> m_nd_item;
};

#endif // ARCANE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme hiérarchique.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors d'Arcane.
 *
 * L'intervalle d'itération est décomposé en \a N WorkGroup contenant chacun \a P WorkItem.
 *
 * \note Sur accélérateur, La valeur de \a P est dépendante de l'architecture
 * de l'accélérateur. Afin d'être portable, cette valeur doit être comprise entre 32 et 1024
 * et être un multiple de 32.
 */
class ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
{
 public:

  //! Type de l'index de la boucle
  using LoopIndexType = WorkGroupLoopContext;

 public:

  //TODO: Faire une méthode makeWorkGroupLoopRange()
  //! Créé un intervalle d'itération pour la command \a command pour \a nb_group de taille \a block_size
  WorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 block_size);

 public:

  //! Nombre d'éléments à traiter
  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return m_total_size; }
  //! Taille d'un groupe
  constexpr ARCCORE_HOST_DEVICE Int32 groupSize() const { return m_group_size; }
  //! Nombre de groupes
  constexpr ARCCORE_HOST_DEVICE Int32 nbGroup() const { return m_nb_group; }

 public:

  //TODO rendre privé ou mettre en externe
#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
  constexpr ARCCORE_HOST_DEVICE WorkGroupLoopContext getIndices(Int32) const { return WorkGroupLoopContext(); }
#endif

#if defined(ARCANE_COMPILING_SYCL)
  //TODO rendre privé ou mettre en externe
  SyclWorkGroupLoopContext getIndices(sycl::nd_item<1> id) const
  {
    return SyclWorkGroupLoopContext(id);
  }
#endif

 private:

  Int32 m_total_size = 0;
  Int32 m_nb_group = 0;
  Int32 m_group_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Applique le fonctor \a func sur une boucle séqentielle.
template <typename Lambda, typename... RemainingArgs> void
arcaneSequentialFor(WorkGroupLoopRange bounds, const Lambda& func, RemainingArgs... remaining_args)
{
  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtBegin(remaining_args...);
  const Int32 group_size = bounds.groupSize();
  const Int32 nb_group = bounds.nbGroup();
  Int32 loop_index = 0;
  for (Int32 i = 0; i < nb_group; ++i) {
    func(WorkGroupLoopContext(loop_index, i, group_size), remaining_args...);
    loop_index += group_size;
  }

  ::Arcane::Impl::HostKernelRemainingArgsHelper::applyRemainingArgsAtEnd(remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique le fonctor \a func sur une boucle parallèle
template <typename Lambda, typename... RemainingArgs> void
arccoreParallelFor(WorkGroupLoopRange bounds,
                   [[maybe_unused]] const ForLoopRunInfo& run_info,
                   const Lambda& func, RemainingArgs... remaining_args)
{
  // Pour l'instant on ne fait que du séquentiel.
  arcaneSequentialFor(bounds, func, remaining_args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour Sycl, le type de l'itérateur ne peut pas être le même sur l'hôte et
// le device car il faut un 'sycl::nd_item' et il n'est pas possible d'en
// construire un (pas de constructeur par défaut). On utilise donc
// une lambda template et le type de l'itérateur est un paramètre template
#if defined(ARCANE_COMPILING_SYCL)
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(auto iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#else
//! Macro pour lancer une commande avec le support du parallélisme hiérarchique
#define RUNCOMMAND_LAUNCH(iter_name, bounds, ...) \
  A_FUNCINFO << ::Arcane::Accelerator::impl::makeExtendedLoop(bounds __VA_OPT__(, __VA_ARGS__)) \
             << [=] ARCCORE_HOST_DEVICE(typename decltype(bounds)::LoopIndexType iter_name __VA_OPT__(ARCANE_RUNCOMMAND_REDUCER_FOR_EACH(__VA_ARGS__)))
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
