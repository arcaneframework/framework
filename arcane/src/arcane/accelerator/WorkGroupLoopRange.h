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

#include "arcane/accelerator/AcceleratorUtils.h"

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
namespace Impl
{
  class WorkGroupSequentialForHelper;
} // namespace Impl

class WorkGroupLoopRange;
class WorkGroupLoopContext;
class HostWorkItemGroup;
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
  friend HostWorkItemGroup;

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
 * \brief Gère un groupe de WorkItem dans un WorkGroupLoopRange pour l'hôte.
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
class HostWorkItemGroup
{
  friend WorkGroupLoopContext;
  friend SyclDeviceWorkItemBlock;
  friend DeviceWorkItemBlock;

 private:

  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE HostWorkItemGroup(Int32 loop_index, Int32 group_index, Int32 group_size, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_group_size(group_size)
  , m_group_index(group_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  constexpr Int32 groupRank() const { return m_group_index; }

  //! Nombre de WorkItem dans un WorkGroup.
  constexpr Int32 groupSize() const { return m_group_size; }

  //! Rang du WorkItem actif dans son WorkGroup.
  constexpr Int32 activeWorkItemRankInGroup() const { return 0; }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return false; }

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  void barrier() {}

  //! Nombre de \a WorkItem à gérer dans l'itération
  constexpr Int32 nbActiveItem() const { return m_nb_active_item; }

  //! Récupère le \a index-ème \a WorkItem à gérer
  WorkItem activeItem(Int32 index) const
  {
    ARCANE_CHECK_AT(index, m_nb_active_item);
    return WorkItem(m_loop_index + index);
  }

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA_OR_HIP)

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
   * \brief Constructeur pour le device.
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_thread_block()
   */
  explicit __device__ DeviceWorkItemBlock()
  : m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  //! Nombre de WorkItem dans un WorkGroup.
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  //! Rang du WorkItem actif dans son WorkGroup.
  __device__ Int32 activeWorkItemRankInGroup() const { return m_thread_block.thread_index().x; }

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  __device__ void barrier() { m_thread_block.sync(); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr __device__ bool isDevice() { return true; }

  //! Nombre de \a WorkItem à gérer dans l'itération
  constexpr __device__ Int32 nbActiveItem() const { return 1; }

  //! Récupère le \a index-ème \a WorkItem à gérer
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
 * Cette classe est utilisée pour l'hôte (séquentiel et multi-thread) et
 * pour CUDA et ROCM/HIP.
 * La méthode group() est différente sur accélérateur et sur l'hôte ce qui
 * permet de particulariser le traitement de la commande.
 */
class WorkGroupLoopContext
{
  // Pour accéder aux constructeurs
  friend WorkGroupLoopRange;
  friend Impl::WorkGroupSequentialForHelper;
  friend constexpr ARCCORE_HOST_DEVICE WorkGroupLoopContext arcaneGetLoopIndexCudaHip(const WorkGroupLoopRange& loop_range, Int32 i);

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  explicit constexpr WorkGroupLoopContext(Int32 loop_index, Int32 group_index, Int32 group_size, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  , m_nb_active_item(nb_active_item)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContext() {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCANE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ DeviceWorkItemBlock group() const { return DeviceWorkItemBlock(); }
#else
  //! Groupe courant
  HostWorkItemGroup group() const { return HostWorkItemGroup(m_loop_index, m_group_index, m_group_size, m_nb_active_item); }
#endif

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
  Int32 m_nb_active_item = 0;
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

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  void barrier() { m_nd_item.barrier(); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return true; }

  //! Nombre de \a WorkItem à gérer dans l'itération
  constexpr Int32 nbActiveItem() const { return 1; }

  //! Récupère le \a index-ème \a WorkItem à gérer
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
 * \brief Contexte d'exécution d'une WorkGroupLoopRange pour Sycl.
 *
 * Cette classe est utilisée uniquement pour la polique d'exécution eAcceleratorPolicy::SYCL.
 */
class SyclWorkGroupLoopContext
{
  friend WorkGroupLoopRange;
  friend SyclWorkGroupLoopContext arcaneGetLoopIndexSycl(const WorkGroupLoopRange& loop_range,
                                                         sycl::nd_item<1> id);

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclWorkGroupLoopContext(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  //! Groupe courant
  SyclDeviceWorkItemBlock group() const { return SyclDeviceWorkItemBlock(m_nd_item); }

 private:

  sycl::nd_item<1> m_nd_item;
};

#endif // ARCANE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme hiérarchique.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de %Arcane.
 *
 * L'intervalle d'itération contient nbElement() et est décomposé en
 * \a nbGroup() WorkGroup contenant chacun \a groupSize() WorkItem.
 *
 * La création de ces instances se fait via les méthodes makeWorkGroupLoopRange().
 *
 * \note Sur accélérateur, La valeur de \a groupSize() est dépendante de l'architecture
 * de l'accélérateur. Afin d'être portable, cette valeur doit être comprise entre 32 et 1024
 * et être un multiple de 32.
 */
class ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
{
 private:

  friend ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
  makeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size);
  friend ARCANE_ACCELERATOR_EXPORT WorkGroupLoopRange
  makeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size);

 public:

  using LoopIndexType = WorkGroupLoopContext;

 public:

  WorkGroupLoopRange() = default;

 private:

  /*!
   * \brief Créé un intervalle d'itération pour la commande \a command.
   *
   * Le nombre total d'éléments est \a total_nb_element, réparti en \a nb_group de taille \a group_size.
   * \a total_nb_element n'est pas nécessairement un multiple de \a block_size.
   */
  WorkGroupLoopRange(Int32 total_nb_element, Int32 nb_group, Int32 group_size);

 public:

  //! Nombre d'éléments à traiter
  constexpr Int32 nbElement() const { return m_total_size; }
  //! Taille d'un groupe
  constexpr Int32 groupSize() const { return m_group_size; }
  //! Nombre de groupes
  constexpr Int32 nbGroup() const { return m_nb_group; }
  //! Nombre d'éléments du dernier groupe
  constexpr Int32 lastGroupSize() const { return m_last_group_size; }
  //! Nombre d'éléments actifs pour le i-ème groupe
  constexpr Int32 nbActiveItem(Int32 i) const
  {
    return ((i + 1) != m_nb_group) ? m_group_size : m_last_group_size;
  }

 private:

  Int32 m_total_size = 0;
  Int32 m_nb_group = 0;
  Int32 m_group_size = 0;
  Int32 m_last_group_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA_OR_HIP)
inline constexpr ARCCORE_HOST_DEVICE WorkGroupLoopContext
arcaneGetLoopIndexCudaHip([[maybe_unused]] const WorkGroupLoopRange& loop_range,
                          [[maybe_unused]] Int32 i)
{
  return WorkGroupLoopContext();
}
#endif

#if defined(ARCANE_COMPILING_SYCL)

namespace Impl
{
  // Pour indiquer qu'il faut toujours utiliser sycl::nd_item (et jamais sycl::id)
  // comme argument avec 'WorkGroupLoopRange.
  template <>
  class IsAlwaysUseSyclNdItem<WorkGroupLoopRange>
  : public std::true_type
  {
  };
} // namespace Impl

inline SyclWorkGroupLoopContext
arcaneGetLoopIndexSycl([[maybe_unused]] const WorkGroupLoopRange& loop_range,
                       sycl::nd_item<1> id)
{
  return SyclWorkGroupLoopContext(id);
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
