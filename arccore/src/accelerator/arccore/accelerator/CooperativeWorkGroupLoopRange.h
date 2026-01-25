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
  template<typename T> friend class CooperativeWorkGroupLoopContext;

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la grille de WorkItem dans un
 * CooperativeWorkGroupLoopRange pour un device CUDA ou HIP.
 */
class CooperativeDeviceWorkItemGrid
{
  template <typename T> friend class CooperativeWorkGroupLoopContext;

 private:

  /*!
   * \brief Constructeur pour le device.
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_grid()
   */
  __device__ CooperativeDeviceWorkItemGrid()
  : m_grid_group(cooperative_groups::this_grid())
  {}

 public:

  //! Bloque tant que tous les \a WorkItem de la grille ne sont pas arrivés ici.
  __device__ void barrier() { m_grid_group.sync(); }

 private:

  cooperative_groups::grid_group m_grid_group;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
template <typename IndexType_>
class CooperativeWorkGroupLoopContext
: public WorkGroupLoopContextBase<IndexType_>
{
  // Pour accéder aux constructeurs
  friend class CooperativeWorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupSequentialForHelper;
  friend Impl::WorkGroupLoopContextBuilder;
  using BaseClass = WorkGroupLoopContextBase<IndexType_>;

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  constexpr CooperativeWorkGroupLoopContext(Int32 loop_index, Int32 group_index,
                                            Int32 group_size, Int32 nb_active_item, Int64 total_size)
  : BaseClass(loop_index, group_index, group_size, nb_active_item, total_size)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE CooperativeWorkGroupLoopContext(Int64 total_size)
  : BaseClass(total_size)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ CooperativeDeviceWorkItemGrid grid() const { return CooperativeDeviceWorkItemGrid{}; }
#else
  //! Groupe courant
  CooperativeHostWorkItemGrid grid() const { return CooperativeHostWorkItemGrid{}; }
#endif
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
class SyclCooperativeDeviceWorkItemGrid
{
  template <typename T> friend class SyclCooperativeWorkGroupLoopContext;

 private:

  explicit SyclCooperativeDeviceWorkItemGrid(sycl::nd_item<1> n)
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
 * Cette classe est utilisée uniquement pour la polique
 * d'exécution eAcceleratorPolicy::SYCL.
 */
template <typename IndexType_>
class SyclCooperativeWorkGroupLoopContext
: public SyclWorkGroupLoopContextBase<IndexType_>
{
  friend CooperativeWorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupLoopContextBuilder;

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclCooperativeWorkGroupLoopContext(sycl::nd_item<1> nd_item, Int64 total_size)
  : SyclWorkGroupLoopContextBase<IndexType_>(nd_item, total_size)
  {
  }

 public:

  //! Grille courante
  SyclCooperativeDeviceWorkItemGrid grid() const
  {
    return SyclCooperativeDeviceWorkItemGrid(this->m_nd_item);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
template <typename IndexType_>
class CooperativeWorkGroupLoopRange
: public WorkGroupLoopRangeBase<IndexType_>
{
 private:

  friend ARCCORE_ACCELERATOR_EXPORT CooperativeWorkGroupLoopRange<Int32>
  makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_group, Int32 group_size);
  friend ARCCORE_ACCELERATOR_EXPORT CooperativeWorkGroupLoopRange<Int32>
  makeCooperativeWorkGroupLoopRange(RunCommand& command, Int32 nb_element, Int32 nb_group, Int32 group_size);

 public:

  using LoopIndexType = CooperativeWorkGroupLoopContext<IndexType_>;
  using IndexType = IndexType_;

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
  CooperativeWorkGroupLoopRange(Int32 total_nb_element, Int32 nb_group, Int32 group_size)
  : WorkGroupLoopRangeBase<IndexType_>(total_nb_element, nb_group, group_size)
  {}

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
