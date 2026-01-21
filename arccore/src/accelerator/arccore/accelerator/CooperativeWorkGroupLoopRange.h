// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CooperativeWorkGroupLoopRange.h                             (C) 2000-2026 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique coopératif.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_COOPERATIVEWORKGROUPLOOPRANGE_H
#define ARCCORE_ACCELERATOR_COOPERATIVEWORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/WorkGroupLoopRange.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
namespace Impl
{
  class CooperativeWorkGroupSequentialForHelper;
} // namespace Impl

class CooperativeWorkGroupLoopRange;
class CooperativeWorkGroupLoopContext;
class HostCooperativeWorkItemGrid;
class SyclDeviceCooperativeWorkItemBlock;
class DeviceCooperativeWorkItemGrid;
class SyclCooperativeWorkGroupLoopContext;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un groupe de WorkItem dans un
 * CooperativeWorkGroupLoopRange pour l'hôte.
 *
 * Le fonctionnement est identique à HostWorkItemGroup.
 * Cette classe possède juste en plus une méthode gridBarrier() qui effectue
 * une barrière sur l'ensemble des threads participants en mode multi-thread.
 */
class HostCooperativeWorkItemGrid
{
  friend CooperativeWorkGroupLoopContext;
  friend SyclDeviceCooperativeWorkItemBlock;
  friend DeviceCooperativeWorkItemGrid;

 private:

  //! Constructeur pour l'hôte
  explicit constexpr ARCCORE_HOST_DEVICE HostCooperativeWorkItemGrid(Int32 loop_index, Int32 group_index, Int32 group_size, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_group_size(group_size)
  , m_group_index(group_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  //! Rang du groupe du CooperativeWorkItem dans la liste des CooperativeWorkGroup.
  constexpr Int32 groupRank() const { return m_group_index; }

  //! Nombre de CooperativeWorkItem dans un CooperativeWorkGroup.
  constexpr Int32 groupSize() const { return m_group_size; }

  //! Rang du CooperativeWorkItem actif dans son CooperativeWorkGroup.
  constexpr Int32 activeCooperativeWorkItemRankInGroup() const { return 0; }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return false; }

  //! Bloque tant que tous les \a WorkItem de la grille ne sont pas arrivés ici.
  void gridBarrier()
  {
    // TODO: A implementer pour le multi-threading via std::barrier()
  }

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  void groupBarrier()
  {
    // Rien à faire car les WorkItem d'un groupe sont exécutés par
    // le même std::thread.
  }

  //! Nombre de \a WorkItem à gérer dans l'itération
  constexpr Int32 nbActiveItem() const { return m_nb_active_item; }

  //! Récupère le \a index-ème \a WorkItem à gérer
  WorkItem activeItem(Int32 index) const
  {
    ARCCORE_CHECK_AT(index, m_nb_active_item);
    return WorkItem(m_loop_index + index);
  }

  constexpr Int32 gridDim() const { return 1; }

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de CooperativeWorkItem dans un
 * CooperativeWorkGroupLoopRange pour un device CUDA ou ROCM.
 */
class DeviceCooperativeWorkItemGroup
{
  friend CooperativeWorkGroupLoopContext;

 private:

  /*!
   * \brief Constructeur pour le device.
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_grid()
   */
  explicit __device__ DeviceCooperativeWorkItemGroup()
  : m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  //! Rang du groupe du CooperativeWorkItem dans la liste des CooperativeWorkGroup.
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  //! Nombre de CooperativeWorkItem dans un CooperativeWorkGroup.
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  //! Rang du CooperativeWorkItem actif dans son CooperativeWorkGroup.
  __device__ Int32 activeCooperativeWorkItemRankInGroup() const { return m_thread_block.thread_index().x; }

  //! Bloque tant que tous les \a CooperativeWorkItem du groupe ne sont pas arrivés ici.
  __device__ void barrier() { m_thread_block.sync(); }

#if 0
  //! Nombre de \a CooperativeWorkItem à gérer dans l'itération
  constexpr __device__ Int32 nbActiveItem() const { return 1; }

  //! Récupère le \a index-ème \a CooperativeWorkItem à gérer
  __device__ CooperativeWorkItem activeItem([[maybe_unused]] Int32 index)
  {
    // Seulement valide pour index==0
    ARCCORE_CHECK_AT(index, 1);
    return CooperativeWorkItem(blockDim.x * blockIdx.x + threadIdx.x);
  }
#endif

 private:

  cooperative_groups::thread_block m_thread_block;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de CooperativeWorkItem dans un
 * CooperativeWorkGroupLoopRange pour un device CUDA ou ROCM.
 */
class DeviceCooperativeWorkItemGrid
{
  friend CooperativeWorkGroupLoopContext;

 private:

  /*!
   * \brief Constructeur pour le device.
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_grid()
   */
  explicit __device__ DeviceCooperativeWorkItemGrid()
  : m_grid_group(cooperative_groups::this_grid())
  , m_thread_block(cooperative_groups::this_thread_block())
  {}

 public:

  //! Rang du groupe du CooperativeWorkItem dans la liste des WorkGroup.
  __device__ Int32 groupRank() const { return m_thread_block.group_index().x; }

  //! Nombre de WorkItem dans un WorkGroup.
  __device__ Int32 groupSize() { return m_thread_block.group_dim().x; }

  //! Rang du WorkItem actif dans son WorkGroup.
  __device__ Int32 activeWorkItemRankInGroup() const { return m_thread_block.thread_index().x; }

  //! Bloque tant que tous les \a CooperativeWorkItem de la grille ne sont pas arrivés ici.
  __device__ void gridBarrier() { m_grid_group.sync(); }

  //! Bloque tant que tous les \a CooperativeWorkItem du groupe ne sont pas arrivés ici.
  __device__ void blockBarrier() { m_thread_block.sync(); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr __device__ bool isDevice() { return true; }

  __device__ Int32 gridDim() const { return m_grid_group.group_dim().x; }
#if 0
  //! Nombre de \a WorkItem à gérer dans l'itération
  constexpr __device__ Int32 nbActiveItem() const { return 1; }

  //! Récupère le \a index-ème \a WorkItem à gérer
  __device__ WorkItem activeItem([[maybe_unused]] Int32 index)
  {
    // Seulement valide pour index==0
    ARCCORE_CHECK_AT(index, 1);
    return WorkItem(blockDim.x * blockIdx.x + threadIdx.x);
  }
#endif

 private:

  cooperative_groups::grid_group m_grid_group;
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
class CooperativeWorkGroupLoopContext
{
  // Pour accéder aux constructeurs
  friend CooperativeWorkGroupLoopRange;
  friend Impl::WorkGroupSequentialForHelper;
  friend constexpr ARCCORE_HOST_DEVICE CooperativeWorkGroupLoopContext
  arcaneGetLoopIndexCudaHip(const CooperativeWorkGroupLoopRange& loop_range);

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  explicit constexpr CooperativeWorkGroupLoopContext(Int32 loop_index, Int32 group_index, Int32 group_size, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  , m_nb_active_item(nb_active_item)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE CooperativeWorkGroupLoopContext() {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ DeviceCooperativeWorkItemGrid group() const { return DeviceCooperativeWorkItemGrid(); }
#else
  //! Groupe courant
  HostCooperativeWorkItemGrid group() const { return HostCooperativeWorkItemGrid(m_loop_index, m_group_index, m_group_size, m_nb_active_item); }
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
#if defined(ARCCORE_COMPILING_SYCL)

/*!
 * \brief Gère un bloc de WorkItem dans un CooperativeWorkGroupLoopRange pour un device Sycl.
 */
class SyclDeviceCooperativeWorkItemGrid
{
  friend SyclCooperativeWorkGroupLoopContext;

 private:

  explicit SyclDeviceCooperativeWorkItemGrid(sycl::nd_item<1> n)
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

  //! Bloque tant que tous les \a CooperativeWorkItem de la grille ne sont pas arrivés ici.
  void groupBarrier() { m_nd_item.barrier(); }

  //! Bloque tant que tous les \a CooperativeWorkItem du groupe ne sont pas arrivés ici.
  void gridBarrier() { /* Not Yet Implemented */ }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return true; }

  //! Nombre de \a CooperativeWorkItem à gérer dans l'itération
  constexpr Int32 nbActiveItem() const { return 1; }

  //! Récupère le \a index-ème \a CooperativeWorkItem à gérer
  WorkItem activeItem(Int32 index)
  {
    // Seulement valide pour index==0
    ARCCORE_CHECK_AT(index, 1);
    return WorkItem(static_cast<Int32>(m_nd_item.get_group(0) * m_nd_item.get_local_range(0) + m_nd_item.get_local_id(0)));
  }

 private:

  sycl::nd_item<1> m_nd_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution d'une CooperativeWorkGroupLoopRange pour Sycl.
 *
 * Cette classe est utilisée uniquement pour la polique d'exécution eAcceleratorPolicy::SYCL.
 */
class SyclCooperativeWorkGroupLoopContext
{
  friend CooperativeWorkGroupLoopRange;
  friend SyclCooperativeWorkGroupLoopContext arcaneGetLoopIndexSycl(const CooperativeWorkGroupLoopRange& loop_range,
                                                                    sycl::nd_item<1> id);

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclCooperativeWorkGroupLoopContext(sycl::nd_item<1> n)
  : m_nd_item(n)
  {
  }

 public:

  //! Groupe courant
  SyclDeviceCooperativeWorkItemGrid group() const { return SyclDeviceCooperativeWorkItemGrid(m_nd_item); }

 private:

  sycl::nd_item<1> m_nd_item;
};

#endif // ARCCORE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme
 * hiérarchique en mode coopératif.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors de %Arcane.
 *
 * L'intervalle d'itération contient nbElement() et est décomposé en
 * \a nbGroup() CooperativeWorkGroup contenant chacun \a groupSize() CooperativeWorkItem.
 *
 * La création de ces instances se fait via les méthodes makeCooperativeWorkGroupLoopRange().
 *
 * \note Sur accélérateur, La valeur de \a groupSize() est dépendante de l'architecture
 * de l'accélérateur. Afin d'être portable, cette valeur doit être comprise entre 32 et 1024
 * et être un multiple de 32.
 */
class ARCCORE_ACCELERATOR_EXPORT CooperativeWorkGroupLoopRange
{
 private:

  friend ARCCORE_ACCELERATOR_EXPORT CooperativeWorkGroupLoopRange
  makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size);
  friend ARCCORE_ACCELERATOR_EXPORT CooperativeWorkGroupLoopRange
  makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size);

 public:

  using LoopIndexType = CooperativeWorkGroupLoopContext;
  // Pour indiquer au KernelLauncher qu'on souhaite un lancement coopératif.
  static constexpr bool isCooperativeLaunch() { return true; }

 public:

  CooperativeWorkGroupLoopRange() = default;

 private:

  /*!
   * \brief Créé un intervalle d'itération pour la commande \a command.
   *
   * Le nombre total d'éléments est \a total_nb_element, réparti en \a nb_group de taille \a group_size.
   * \a total_nb_element n'est pas nécessairement un multiple de \a block_size.
   */
  CooperativeWorkGroupLoopRange(Int32 total_nb_element, Int32 nb_group, Int32 group_size);

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

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
