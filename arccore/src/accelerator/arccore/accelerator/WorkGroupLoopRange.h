// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* WorkGroupLoopRange.h                                        (C) 2000-2026 */
/*                                                                           */
/* Boucle pour le parallélisme hiérarchique.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_WORKGROUPLOOPRANGE_H
#define ARCCORE_ACCELERATOR_WORKGROUPLOOPRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "AcceleratorGlobal.h"
#include "arccore/accelerator/AcceleratorUtils.h"

#if defined(ARCCORE_COMPILING_CUDA)
#include <cooperative_groups.h>
#endif
#if defined(ARCCORE_COMPILING_HIP)
#include <hip/hip_cooperative_groups.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
namespace Impl
{
  class WorkGroupLoopContextBuilder;
  class WorkGroupSequentialForHelper;
} // namespace Impl

template <typename IndexType_ = Int32>
class HostWorkItem;
template <typename IndexType_ = Int32>
class DeviceWorkItem;
template <typename IndexType_ = Int32>
class SyclDeviceWorkItem;

template <typename IndexType_ = Int32>
class WorkGroupLoopRange;
template <typename IndexType_ = Int32>
class CooperativeWorkGroupLoopRange;

template <typename IndexType_ = Int32>
class WorkGroupLoopContext;
template <typename IndexType_ = Int32>
class CooperativeWorkGroupLoopContext;
template <typename IndexType_ = Int32>
class SyclWorkGroupLoopContext;
template <typename IndexType_ = Int32>
class SyclCooperativeWorkGroupLoopContext;

class HostWorkItemBlock;
class SyclDeviceWorkItemBlock;
class DeviceWorkItemBlock;

class CooperativeHostWorkItemGrid;
class SyclDeviceCooperativeWorkItemGrid;

template <typename Indextype_ = Int32>
class WorkGroupLoopContextBase;
template <typename Indextype_ = Int32>
class SyclWorkGroupLoopContextBase;

template <typename IndexType_ = Int32>
class HostIndexes;
template <typename IndexType_ = Int32>
class DeviceIndexesBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class HostIndexes
{
 public:

  using IndexType = IndexType_;

  class HostWorkItemIterator
  {
   public:

    explicit constexpr HostWorkItemIterator(IndexType loop_index)
    : m_loop_index(loop_index)
    {}
    constexpr IndexType operator*() const { return m_loop_index; }
    HostWorkItemIterator& operator++()
    {
      ++m_loop_index;
      return (*this);
    }
    friend bool operator!=(HostWorkItemIterator a, HostWorkItemIterator b)
    {
      return a.m_loop_index != b.m_loop_index;
    }

   private:

    IndexType m_loop_index = 0;
  };

 public:

  constexpr HostIndexes(IndexType loop_index, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  constexpr HostWorkItemIterator begin() const { return HostWorkItemIterator(m_loop_index); }
  constexpr HostWorkItemIterator end() const { return HostWorkItemIterator(m_loop_index + m_nb_active_item); }

 private:

  IndexType m_loop_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class DeviceIndexesBase
{
 public:

  using IndexType = IndexType_;

  class DeviceWorkItemIterator
  {
   public:

    explicit constexpr DeviceWorkItemIterator(IndexType loop_index, Int32 grid_size)
    : m_loop_index(loop_index)
    , m_grid_size(grid_size)
    {}
    constexpr IndexType operator*() const { return m_loop_index; }
    ARCCORE_HOST_DEVICE DeviceWorkItemIterator& operator++()
    {
      m_loop_index += m_grid_size;
      return (*this);
    }
    friend constexpr bool operator!=(DeviceWorkItemIterator a, DeviceWorkItemIterator b)
    {
      return a.m_loop_index != b.m_loop_index;
    }

   private:

    IndexType m_loop_index = 0;
    Int32 m_grid_size = 0;
  };

  class DeviceWorkItemSentinel
  {
   public:

    explicit constexpr DeviceWorkItemSentinel(IndexType total_size)
    : m_total_size(total_size)
    {}
    friend constexpr bool operator!=(DeviceWorkItemIterator a, DeviceWorkItemSentinel b)
    {
      return *a < b.m_total_size;
    }

   private:

    IndexType m_total_size = 0;
  };
};

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

template <typename IndexType_>
class DeviceIndexes
: public DeviceIndexesBase<IndexType_>
{
 public:

  using IndexType = IndexType_;
  using DeviceWorkItemIterator = DeviceIndexesBase<IndexType_>::DeviceWorkItemIterator;
  using DeviceWorkItemSentinel = DeviceIndexesBase<IndexType_>::DeviceWorkItemSentinel;

 public:

  explicit constexpr DeviceIndexes(IndexType total_size)
  : m_total_size(total_size)
  {}

 public:

  __device__ DeviceWorkItemIterator begin() const
  {
    return DeviceWorkItemIterator(blockDim.x * blockIdx.x + threadIdx.x, blockDim.x * gridDim.x);
  }
  constexpr __device__ DeviceWorkItemSentinel end() const
  {
    return DeviceWorkItemSentinel(m_total_size);
  }

 private:

  IndexType m_total_size = 0;
};

#endif

#if defined(ARCCORE_COMPILING_SYCL)

template <typename IndexType_>
class SyclDeviceIndexes
: public DeviceIndexesBase<IndexType_>
{
 public:

  using IndexType = IndexType_;
  using DeviceWorkItemIterator = DeviceIndexesBase<IndexType_>::DeviceWorkItemIterator;
  using DeviceWorkItemSentinel = DeviceIndexesBase<IndexType_>::DeviceWorkItemSentinel;

 public:

  SyclDeviceIndexes(sycl::nd_item<1> nd_item, IndexType total_size)
  : m_nd_item(nd_item)
  , m_total_size(total_size)
  {}

 public:

  DeviceWorkItemIterator begin() const
  {
    IndexType index = static_cast<IndexType>(m_nd_item.get_group(0) * m_nd_item.get_local_range(0) + m_nd_item.get_local_id(0));
    Int32 grid_size = static_cast<Int32>(m_nd_item.get_local_range(0) * m_nd_item.get_group_range(0));
    return DeviceWorkItemIterator(index, grid_size);
  }
  constexpr DeviceWorkItemSentinel end() const { return DeviceWorkItemSentinel(m_total_size); }

 private:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère pour l'hôte un WorkItem dans un WorkGroupLoopRange ou
 * CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class HostWorkItem
{
  template <typename T> friend class WorkGroupLoopContextBase;

 public:

  using IndexType = IndexType_;

 private:

  //! Constructeur pour l'hôte
  constexpr ARCCORE_HOST_DEVICE HostWorkItem(IndexType loop_index, Int32 nb_active_item)
  : m_loop_index(loop_index)
  , m_nb_active_item(nb_active_item)
  {}

 public:

  //! Rang du WorkItem actif dans son WorkGroup.
  constexpr Int32 rankInBlock() const { return 0; }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return false; }

  //! Indexes de la boucle gérés par ce WorkItem
  constexpr HostIndexes<IndexType> linearIndexes() const
  {
    return HostIndexes<IndexType>(m_loop_index, m_nb_active_item);
  }

 private:

  IndexType m_loop_index = 0;
  Int32 m_nb_active_item = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère pour l'hôte un groupe de WorkItem dans un WorkGroupLoopRange
 * ou un CooperativeWorkGroupLoopRange.
 */
class HostWorkItemBlock
{
  template <typename T> friend class WorkGroupLoopContextBase;

 private:

  //! Constructeur pour l'hôte
  constexpr ARCCORE_HOST_DEVICE HostWorkItemBlock(Int32 group_index, Int32 group_size)
  : m_group_size(group_size)
  , m_group_index(group_index)
  {}

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  constexpr Int32 groupRank() const { return m_group_index; }

  //! Nombre de WorkItem dans un WorkGroup.
  constexpr Int32 groupSize() const { return m_group_size; }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return false; }

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  void barrier() {}

 private:

  Int32 m_group_size = 0;
  Int32 m_group_index = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère pour un device CUDA ou HIP un WorkItem dans un
 * WorkGroupLoopRange ou un CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class DeviceWorkItem
{
  friend class WorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  /*!
   * \brief Constructeur pour le device.
   *
   * Ce constructeur n'a pas besoin d'informations spécifiques car tout est
   * récupéré via cooperative_groups::this_thread_block()
   */
  explicit __device__ DeviceWorkItem(IndexType total_size)
  : m_thread_block(cooperative_groups::this_thread_block())
  , m_total_size(total_size)
  {}

 public:

  //! Rang du WorkItem dans son WorkGroup.
  __device__ Int32 rankInBlock() const { return m_thread_block.thread_index().x; }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr __device__ bool isDevice() { return true; }

  constexpr __device__ DeviceIndexes<IndexType> linearIndexes() const
  {
    return DeviceIndexes<IndexType>(m_total_size);
  }

 private:

  // TODO A supprimer
  cooperative_groups::thread_block m_thread_block;
  IndexType_ m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de WorkItem dans un WorkGroupLoopRange pour un device CUDA ou ROCM.
 */
class DeviceWorkItemBlock
{
  template <typename T> friend class WorkGroupLoopContextBase;

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

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  __device__ void barrier() { m_thread_block.sync(); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr __device__ bool isDevice() { return true; }

 private:

  cooperative_groups::thread_block m_thread_block;
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour WorkGroupLoopContext et CooperativeWorkGroupLoopContext.
 */
template <typename IndexType_>
class WorkGroupLoopContextBase
{
 public:

  using IndexType = IndexType_;

 protected:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  constexpr WorkGroupLoopContextBase(IndexType loop_index, Int32 group_index, Int32 group_size,
                                     Int32 nb_active_item, IndexType total_size)
  : m_loop_index(loop_index)
  , m_total_size(total_size)
  , m_group_index(group_index)
  , m_group_size(group_size)
  , m_nb_active_item(nb_active_item)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContextBase(IndexType total_size)
  : m_total_size(total_size)
  {}

 public:

#if defined(ARCCORE_DEVICE_CODE) && !defined(ARCCORE_COMPILING_SYCL)
  //! Groupe courant. Pour CUDA/ROCM, il s'agit d'un bloc de threads.
  __device__ DeviceWorkItemBlock block() const { return DeviceWorkItemBlock(); }
  //! WorkItem actif. Pour CUDA/ROCM, il s'agit d'un thread.
  __device__ DeviceWorkItem<IndexType> workItem() const { return DeviceWorkItem<IndexType>(m_total_size); }
#else
  //! Groupe courant
  HostWorkItemBlock block() const { return HostWorkItemBlock(m_group_index, m_group_size); }
  //! WorkItem actif
  HostWorkItem<IndexType> workItem() const { return { m_loop_index, m_nb_active_item }; }
#endif

 protected:

  IndexType m_loop_index = 0;
  IndexType m_total_size = 0;
  Int32 m_group_index = 0;
  Int32 m_group_size = 0;
  Int32 m_nb_active_item = 0;
};

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
class WorkGroupLoopContext
: public WorkGroupLoopContextBase<IndexType_>
{
  // Pour accéder aux constructeurs
  template <typename T> friend class WorkGroupLoopRange;
  friend Impl::WorkGroupSequentialForHelper;
  friend Impl::WorkGroupLoopContextBuilder;
  using BaseClass = WorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  //! Ce constructeur est utilisé dans l'implémentation hôte.
  explicit constexpr WorkGroupLoopContext(IndexType loop_index, Int32 group_index, Int32 group_size,
                                          Int32 nb_active_item, IndexType total_size,
                                          [[maybe_unused]] Int32 nb_block)
  : BaseClass(loop_index, group_index, group_size, nb_active_item, total_size)
  {
  }

  // Ce constructeur n'est utilisé que sur le device
  // Il ne fait rien car les valeurs utiles sont récupérées via cooperative_groups::this_thread_block()
  explicit constexpr ARCCORE_DEVICE WorkGroupLoopContext(IndexType total_size)
  : BaseClass(total_size)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCCORE_COMPILING_SYCL)

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère pour un device Sycl un WorkItem dans un WorkGroupLoopRange
 * ou un CooperativeWorkGroupLoopRange.
 */
template <typename IndexType_>
class SyclDeviceWorkItem
{
  friend SyclWorkGroupLoopContextBase<IndexType_>;

 public:

  using IndexType = IndexType_;

 private:

  explicit SyclDeviceWorkItem(sycl::nd_item<1> nd_item, IndexType total_size)
  : m_nd_item(nd_item)
  , m_total_size(total_size)
  {
  }

 public:

  //! Rang du WorkItem actif dans le WorkGroup.
  Int32 rankInBlock() const { return static_cast<Int32>(m_nd_item.get_local_id(0)); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return true; }

  SyclDeviceIndexes<IndexType> linearIndexes() const
  {
    return SyclDeviceIndexes<IndexType>(m_nd_item, m_total_size);
  }

 private:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère un bloc de WorkItem dans un WorkGroupLoopRange pour un device Sycl.
 */
class SyclDeviceWorkItemBlock
{
  template <typename T> friend class SyclWorkGroupLoopContextBase;

 private:

  explicit SyclDeviceWorkItemBlock(sycl::nd_item<1> nd_item)
  : m_nd_item(nd_item)
  {
  }

 public:

  //! Rang du groupe du WorkItem dans la liste des WorkGroup.
  Int32 groupRank() const { return static_cast<Int32>(m_nd_item.get_group(0)); }

  //! Nombre de WorkItem dans un WorkGroup.
  Int32 groupSize() { return static_cast<Int32>(m_nd_item.get_local_range(0)); }

  //! Bloque tant que tous les \a WorkItem du groupe ne sont pas arrivés ici.
  void barrier() { m_nd_item.barrier(); }

  //! Indique si on s'exécute sur un accélérateur
  static constexpr bool isDevice() { return true; }

 private:

  sycl::nd_item<1> m_nd_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution d'un WorkGroupLoopRange pour Sycl.
 *
 * Cette classe est utilisée uniquement pour la polique
 * d'exécution eAcceleratorPolicy::SYCL.
 */
template <typename IndexType_>
class SyclWorkGroupLoopContextBase
{
  friend WorkGroupLoopRange<IndexType_>;

 public:

  using IndexType = IndexType_;

 protected:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclWorkGroupLoopContextBase(sycl::nd_item<1> n, IndexType total_size)
  : m_nd_item(n)
  , m_total_size(total_size)
  {
  }

 public:

  //! Groupe courant
  SyclDeviceWorkItemBlock block() const { return SyclDeviceWorkItemBlock(m_nd_item); }

  //! WorkItem courant
  SyclDeviceWorkItem<IndexType_> workItem() const
  {
    return SyclDeviceWorkItem<IndexType_>(m_nd_item, m_total_size);
  }

 protected:

  sycl::nd_item<1> m_nd_item;
  IndexType m_total_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte d'exécution d'un WorkGroupLoopRange pour Sycl.
 *
 * Cette classe est utilisée uniquement pour la polique
 * d'exécution eAcceleratorPolicy::SYCL.
 */
template <typename IndexType_>
class SyclWorkGroupLoopContext
: public SyclWorkGroupLoopContextBase<IndexType_>
{
  friend WorkGroupLoopRange<IndexType_>;
  friend Impl::WorkGroupLoopContextBuilder;

 public:

  using IndexType = IndexType_;

 private:

  // Ce constructeur n'est utilisé que sur le device
  explicit SyclWorkGroupLoopContext(sycl::nd_item<1> nd_item, IndexType total_size)
  : SyclWorkGroupLoopContextBase<IndexType_>(nd_item, total_size)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCCORE_COMPILING_SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme hiérarchique.
 *
 * Cette classe est la classe de base pour WorkGroupLoopRange et CooperativeWorkGroupLoopRange.
 *
 * Il faudra appeler setBlockSize() pour positionner la taille d'un bloc.
 * Cela peut être fait par le développeur ou automatiquement au lancement de la
 * commande.
 *
 * L'intervalle d'itération contient nbElement() et est décomposé en
 * \a nbBlock() WorkGroup contenant chacun \a blockSize() WorkItem.
 *
 * \note Sur accélérateur, La valeur de \a blockSize() est dépendante de l'architecture
 * de l'accélérateur. Afin d'être portable, cette valeur doit être comprise entre 32 et 1024
 * et être un multiple de 32.
 *
 */
template <bool IsCooperativeLaunch, typename IndexType_>
class WorkGroupLoopRangeBase
{
 public:

  using IndexType = IndexType_;

 public:

  WorkGroupLoopRangeBase() = default;
  explicit WorkGroupLoopRangeBase(IndexType nb_element)
  : m_nb_element(nb_element)
  {
  }

 public:

  static constexpr bool isCooperativeLaunch() { return IsCooperativeLaunch; }

  //! Nombre d'éléments à traiter
  constexpr IndexType nbElement() const { return m_nb_element; }
  //! Taille d'un block
  constexpr IndexType blockSize() const { return m_block_size; }
  /*!
   * \brief Nombre de blocs.
   *
   * Retourne 0 si setBlockSize() n'a pas encore été appelé.
   */
  constexpr Int32 nbBlock() const { return m_nb_block; }

  /*!
   * \brief Positionne la taille d'un bloc.
   *
   * \a nb_block doit être un multiple de 32.
   */
  ARCCORE_ACCELERATOR_EXPORT void setBlockSize(IndexType nb_block);

  //! Positionne la taille d'un bloc en fonction de la commande \a command
  ARCCORE_ACCELERATOR_EXPORT void setBlockSize(RunCommand& command);

 private:

  IndexType m_nb_element = 0;
  IndexType m_block_size = 0;
  Int32 m_nb_block = 0;

 private:

  void _setNbBlock();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Intervalle d'itération d'une boucle utilisant le parallélisme hiérarchique.
 *
 * \sa WorkGroupLoopRangeBase
 */
template <typename IndexType_>
class WorkGroupLoopRange
: public WorkGroupLoopRangeBase<false, IndexType_>
{
 public:

  using LoopIndexType = WorkGroupLoopContext<IndexType_>;
  using IndexType = IndexType_;

 public:

  WorkGroupLoopRange() = default;
  explicit WorkGroupLoopRange(IndexType total_nb_element)
  : WorkGroupLoopRangeBase<false, IndexType_>(total_nb_element)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
