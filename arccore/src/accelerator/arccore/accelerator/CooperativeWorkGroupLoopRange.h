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
  explicit CooperativeHostWorkItemGrid(Int32 nb_block)
  : m_nb_block(nb_block)
  {}

 public:

  //! Nombre de blocs dans la grille
  Int32 nbBlock() const { return m_nb_block; }

  //! Bloque tant que tous les \a WorkItem de la grille ne sont pas arrivés ici.
  void barrier()
  {
    // TODO: A implementer pour le multi-threading via std::barrier()
  }

 private:

  Int32 m_nb_block = 0;
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

  //! Nombre de blocs dans la grille
  __device__ Int32 nbBlock() const { return m_grid_group.group_dim().x; }

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

 public:

  using IndexType = IndexType_;

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  constexpr CooperativeWorkGroupLoopContext(IndexType loop_index, Int32 group_index,
                                            Int32 group_size, Int32 nb_active_item,
                                            IndexType total_size, Int32 nb_block)
  : BaseClass(loop_index, group_index, group_size, nb_active_item, total_size)
  , m_nb_block(nb_block)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE CooperativeWorkGroupLoopContext(IndexType total_size, Int32 nb_block)
  : BaseClass(total_size)
  , m_nb_block(nb_block)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ CooperativeDeviceWorkItemGrid grid() const { return CooperativeDeviceWorkItemGrid{}; }
#else
  //! Groupe courant
  CooperativeHostWorkItemGrid grid() const { return CooperativeHostWorkItemGrid(m_nb_block); }
#endif

 private:

  Int32 m_nb_block = 0;
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

  //! Nombre de blocs dans la grille
  Int32 nbBlock() const { return static_cast<Int32>(m_nd_item.get_group_range(0)); }

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

 public:

  using IndexType = IndexType_;

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclCooperativeWorkGroupLoopContext(sycl::nd_item<1> nd_item, IndexType total_size)
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
 * hiérarchique collaboratif.
 *
 * \sa WorkGroupLoopRangeBase
 */
template <typename IndexType_>
class CooperativeWorkGroupLoopRange
: public WorkGroupLoopRangeBase<true, IndexType_>
{
 public:

  using LoopIndexType = CooperativeWorkGroupLoopContext<IndexType_>;
  using IndexType = IndexType_;

 public:

  CooperativeWorkGroupLoopRange() = default;
  explicit CooperativeWorkGroupLoopRange(IndexType total_nb_element)
  : WorkGroupLoopRangeBase<true, IndexType_>(total_nb_element)
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
