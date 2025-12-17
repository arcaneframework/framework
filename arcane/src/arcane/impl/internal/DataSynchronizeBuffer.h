// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeBuffer.h                                     (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un buffer générique pour la synchronisation de données.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZEBUFFER_H
#define ARCANE_IMPL_DATASYNCHRONIZEBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/impl/IDataSynchronizeBuffer.h"
#include "arcane/utils/FixedArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class IBufferCopier;
class DataSynchronizeResult;
class DataSynchronizeInfo;
class DataSynchronizeBufferInfoList;
class MemoryBuffer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base de l'implémentation de IDataSynchronizeBuffer.
 *
 * Cette implémentation utilise un seul buffer mémoire pour gérer les trois
 * parties de la synchronisation : le buffer d'envoi, le buffer de réception
 * et le buffer pour les comparer si la synchronisation a modifié des valeurs
 * (ce dernier est optionnel).
 * Chaque buffer est ensuite séparé en N parties, appelées sous-buffer,
 * avec N le nombre de rangs qui communiquent. Enfin, chaque sous-buffer est
 * lui-même séparé en P parties, avec P le nombre de données à communiquer.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeBufferBase
: public IDataSynchronizeBuffer
{
  /*!
   * \brief Buffer pour un élément de la synchronisation (envoi, réception ou comparaison)
   */
  class BufferInfo
  {
   public:

    //! Buffer global
    MutableMemoryView globalBuffer() const { return m_memory_view; }

    //! Positionne le buffer global.
    void setGlobalBuffer(MutableMemoryView v);

    //! Buffer pour le \a index-ème rang
    MutableMemoryView localBuffer(Int32 rank_index) const;

    //! Buffer pour le \a index-ème rang et la \a data_index-ème donnée
    MutableMemoryView dataLocalBuffer(Int32 rank_index, Int32 data_index) const;

    //! Déplacement dans \a globalBuffer() pour le \a index-ème rang
    Int64 displacement(Int32 rank_index) const;

    //! Taille (en octet) du buffer local pour le rang \a rank_index.
    Int64 localBufferSize(Int32 rank_index) const;

    //! Taille totale en octet du buffer global
    Int64 totalSize() const { return m_total_size; }

    //! Numéros locaux des entités pour le rang \a index
    ConstArrayView<Int32> localIds(Int32 index) const;

    void checkValid() const
    {
      ARCANE_CHECK_POINTER(m_buffer_info);
    }

    void initialize(ConstArrayView<Int32> datatype_sizes, const DataSynchronizeBufferInfoList* buffer_info);

   private:

    /*!
     * \brief Vue sur la zone mémoire du buffer.
     *
     * Cette variable n'est valide qu'après allocation de tous les buffers.
     */
    MutableMemoryView m_memory_view;
    //! Offset (en octet) dans globalBuffer() de chaque donnée
    UniqueArray2<Int64> m_displacements;
    //! Taille (en octet) de chaque buffer local.
    SmallArray<Int64> m_local_buffer_size;
    //! Taille (en octet) du type de chaque donnée.
    ConstArrayView<Int32> m_datatype_sizes;
    //! Taille total (en octet) du buffer
    Int64 m_total_size = 0;
    const DataSynchronizeBufferInfoList* m_buffer_info = nullptr;
  };

 public:

  Int32 nbRank() const final { return m_nb_rank; }
  Int32 targetRank(Int32 index) const final;
  bool hasGlobalBuffer() const final { return true; }

  MutableMemoryView receiveBuffer(Int32 index) final { return m_ghost_buffer_info.localBuffer(index); }
  MutableMemoryView sendBuffer(Int32 index) final { return m_share_buffer_info.localBuffer(index); }

  Int64 receiveDisplacement(Int32 index) const final { return m_ghost_buffer_info.displacement(index); }
  Int64 sendDisplacement(Int32 index) const final { return m_share_buffer_info.displacement(index); }

  MutableMemoryView globalReceiveBuffer() final { return m_ghost_buffer_info.globalBuffer(); }
  MutableMemoryView globalSendBuffer() final { return m_share_buffer_info.globalBuffer(); }

  Int64 totalReceiveSize() const final { return m_ghost_buffer_info.totalSize(); }
  Int64 totalSendSize() const final { return m_share_buffer_info.totalSize(); }

  void barrier() final;

 public:

  DataSynchronizeBufferBase(DataSynchronizeInfo* sync_info, Ref<IBufferCopier> copier);

 public:

  //! Indique si on compare les valeurs avant/après la synchronisation
  bool isCompareSynchronizedValues() const { return m_is_compare_sync_values; }

  void setSynchronizeBuffer(Ref<MemoryBuffer> v)
  {
    m_memory = v;
  }

  /*!
   * \brief Prépare la synchronisation.
   *
   * Prépare la synchronisation et alloue les buffers si nécessaire.
   *
   * Si \a is_compare_sync est vrai, on compare après la synchronisation les
   * valeurs des entités fantômes avec leur valeur d'avant la synchronisation.
   *
   * Il faut avoir appelé setSynchronizeBuffer() au moins une fois avant d'appeler
   * cette méthode pour positionner la zone mémoire allouée.
   */
  virtual void prepareSynchronize(bool is_compare_sync) = 0;

 protected:

  void _allocateBuffers();
  //! Calcule les informations pour la synchronisation
  void _compute(ConstArrayView<Int32> datatype_sizes);

 protected:

  DataSynchronizeInfo* m_sync_info = nullptr;
  //! Buffer pour toutes les données des entités fantômes qui serviront en réception
  BufferInfo m_ghost_buffer_info;
  //! Buffer pour toutes les données des entités partagées qui serviront en envoi
  BufferInfo m_share_buffer_info;
  //! Buffer pour tester si la synchronisation a modifié les valeurs des mailles fantômes
  BufferInfo m_compare_sync_buffer_info;

 protected:

  Int32 m_nb_rank = 0;
  bool m_is_compare_sync_values = false;

  //! Buffer contenant les données concaténées en envoi et réception
  Ref<MemoryBuffer> m_memory;

  Ref<IBufferCopier> m_buffer_copier;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour une donnée
 */
class ARCANE_IMPL_EXPORT SingleDataSynchronizeBuffer
: public TraceAccessor
, public DataSynchronizeBufferBase
{
 public:

  SingleDataSynchronizeBuffer(ITraceMng* tm, DataSynchronizeInfo* sync_info, Ref<IBufferCopier> copier)
  : TraceAccessor(tm)
  , DataSynchronizeBufferBase(sync_info, copier)
  {}

 public:

  void copyReceiveAsync(Int32 index) final;
  void copySendAsync(Int32 index) final;

 public:

  void setDataView(MutableMemoryView v)
  {
    m_data_view = v;
    m_datatype_sizes[0] = v.datatypeSize();
  }
  //! Zone mémoire contenant les valeurs de la donnée à synchroniser
  MutableMemoryView dataView() { return m_data_view; }
  void prepareSynchronize(bool is_compare_sync) override;

  /*!
   * \brief Termine la synchronisation.
   */
  DataSynchronizeResult finalizeSynchronize();

 private:

  //! Vue sur les données de la variable
  MutableMemoryView m_data_view;
  //! Tableau contenant les tailles des types de donnée
  FixedArray<Int32, 1> m_datatype_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IDataSynchronizeBuffer pour plusieurs données.
 */
class ARCANE_IMPL_EXPORT MultiDataSynchronizeBuffer
: public TraceAccessor
, public DataSynchronizeBufferBase
{

 public:

  MultiDataSynchronizeBuffer(ITraceMng* tm, DataSynchronizeInfo* sync_info,
                             Ref<IBufferCopier> copier)
  : TraceAccessor(tm)
  , DataSynchronizeBufferBase(sync_info, copier)
  {}

 public:

  void copyReceiveAsync(Int32 rank_index) final;
  void copySendAsync(Int32 rank_index) final;

 public:

  void setNbData(Int32 nb_data)
  {
    m_data_views.resize(nb_data);
    m_datatype_sizes.resize(nb_data);
  }
  void setDataView(Int32 index, MutableMemoryView v)
  {
    m_data_views[index] = v;
    m_datatype_sizes[index] = v.datatypeSize();
  }

  void prepareSynchronize(bool is_compare_sync) override;

 private:

  //! Vue sur les données de la variable
  SmallArray<MutableMemoryView> m_data_views;
  //! Tableau contenant les tailles des types de donnée
  SmallArray<Int32> m_datatype_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
