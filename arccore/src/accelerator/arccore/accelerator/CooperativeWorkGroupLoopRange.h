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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère une grille de WorkItem dans un
 * CooperativeWorkGroupLoopRange pour l'hôte.
 *
 * Cette classe possède juste une méthode barrier() qui effectue
 * une barrière sur l'ensemble des threads participants en mode multi-thread.
 */
class CooperativeHostWorkItemGrid
{
  friend CooperativeWorkGroupLoopContext;

 private:

  //! Constructeur pour l'hôte
  explicit CooperativeHostWorkItemGrid()
  {}

 public:

  //! Bloque tant que tous les \a WorkItem de la grille ne sont pas arrivés ici.
  void barrier()
  {
    // TODO: A implementer pour le multi-threading via std::barrier()
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la grille de CooperativeWorkItem dans un
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
  {}

 public:

  //! Bloque tant que tous les \a CooperativeWorkItem de la grille ne sont pas arrivés ici.
  __device__ void barrier() { m_grid_group.sync(); }

 private:

  cooperative_groups::grid_group m_grid_group;
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
  explicit constexpr CooperativeWorkGroupLoopContext(Int32 loop_index, Int32 group_index, Int32 group_size, Int32 nb_active_item, Int64 total_size)
  : m_loop_index(loop_index)
  , m_group_index(group_index)
  , m_group_size(group_size)
  , m_nb_active_item(nb_active_item)
  , m_total_size(total_size)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE CooperativeWorkGroupLoopContext(Int64 total_size)
  : m_total_size(total_size)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ DeviceCooperativeWorkItemGrid grid() const { return DeviceCooperativeWorkItemGrid{}; }
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ DeviceWorkItemBlock group() const { return DeviceWorkItemBlock(m_total_size); }
#else
  //! Groupe courant
  CooperativeHostWorkItemGrid grid() const { return CooperativeHostWorkItemGrid{}; }
  //! Groupe courant
  HostWorkItemGroup group() const { return HostWorkItemGroup(m_loop_index, m_group_index, m_group_size, m_nb_active_item); }
#endif

 private:

  Int32 m_loop_index = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
  Int32 m_nb_active_item = 0;
  Int64 m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Implémentation pour SYCL.
 */
#if defined(ARCCORE_COMPILING_SYCL)

/*!
 * \brief Gère la grille de WorkItem dans un CooperativeWorkGroupLoopRange pour un device Sycl.
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

  //! Bloque tant que tous les \a CooperativeWorkItem de la grille ne sont pas arrivés ici.
  void barrier() { /* Not Yet Implemented */ }

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
  explicit SyclCooperativeWorkGroupLoopContext(sycl::nd_item<1> nd_item, Int64 total_size)
  : m_nd_item(nd_item)
  , m_total_size(total_size)
  {
  }

 public:

  //! Grille courante
  SyclDeviceCooperativeWorkItemGrid grid() const { return SyclDeviceCooperativeWorkItemGrid(m_nd_item); }

  //! Groupe courant
  SyclDeviceWorkItemBlock group() const { return SyclDeviceWorkItemBlock(m_nd_item, m_total_size); }

 private:

  sycl::nd_item<1> m_nd_item;
  Int64 m_total_size = 0;
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
