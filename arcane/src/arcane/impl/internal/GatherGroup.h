// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherGroup.h                                               (C) 2000-2026 */
/*                                                                           */
/* Classe permettant de gérer les regroupements de données sur le ou les     */
/* sous-domaines écrivains.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_IMPL_INTERNAL_GATHERGROUP_H
#define ARCANE_IMPL_INTERNAL_GATHERGROUP_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/IGatherGroup.h"

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GatherGroup;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de calculer et de conserver les informations de
 * regroupements.
 *
 * Les écrivains seront les masterIO ou les masterParallelIO si
 * \a m_use_collective_io est vrai.
 */
class ARCANE_IMPL_EXPORT GatherGroupInfo
: public IGatherGroupInfo
{
  friend GatherGroup;

 public:

  /*!
   * \brief Constructeur.
   *
   * \param pm Le parallelMng qui contient les masterIO.
   * \param use_collective_io True si l'on souhaite que tous les processus
   * écrivent (avec MPI-IO par exemple). Les écrivains seront donc les
   * masterParallelIO. Si False, l'écrivain sera masterIO.
   */
  GatherGroupInfo(IParallelMng* pm, bool use_collective_io);

  ~GatherGroupInfo() override;

 public:

  void computeSize(Int32 nb_elem_in) override;
  void needRecompute() override { m_is_computed = false; }
  bool isComputed() override { return m_is_computed; }
  Int32 nbElemOutput() override { return m_nb_elem_output; }
  Int32 sizeOfOutput(Int32 sizeof_type) override { return m_nb_elem_output * sizeof_type; }
  SmallSpan<Int32> nbElemRecvGatherToMasterIO() override;
  Int32 nbWriterGlobal() override { return m_nb_writer_global; }

 public:

  void setCollectiveIO(bool enable) { m_use_collective_io = enable; }

  /*!
   * \brief Méthode permettant de calculer les informations de regroupements.
   *
   * Appel collectif.
   *
   * Un second appel à cette méthode n'aura pas d'effet, sauf en cas d'appel à
   * la méthode \a needRecompute() avant.
   *
   * \param in La vue qui sera partagée.
   */
  template <class T>
  void computeSizeT(Span<const T> in);

  /*!
   * \brief Méthode permettant de calculer les informations de regroupements.
   *
   * Appel collectif.
   *
   * Un second appel à cette méthode n'aura pas d'effet, sauf en cas d'appel à
   * la méthode \a needRecompute() avant.
   *
   * \param in La vue qui sera partagée.
   */
  template <class T>
  void computeSizeT(Span2<const T> in);

 private:

  IParallelMng* m_pm = nullptr;
  bool m_use_collective_io = false;
  UniqueArray<Int32> m_nb_elem_recv;
  Int32 m_nb_elem_output = -1;
  Int32 m_writer = -1;
  Int32 m_nb_sender_to_writer = -1;
  Int32 m_nb_writer_global = -1;
  bool m_is_computed = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de regrouper les données de certains
 * sous-domaines sur d'autres sous-domaines.
 *
 * Les écrivains seront les masterIO ou les masterParallelIO si
 * \a m_use_collective_io est vrai.
 */
class ARCANE_IMPL_EXPORT GatherGroup
: public IGatherGroup
{

 public:

  /*!
   * \brief Constructeur.
   * \param ggi Les informations de regroupement. \a isComputed() devra être
   * vrai.
   */
  explicit GatherGroup(GatherGroupInfo* ggi);

  /*!
   * \brief Constructeur.
   * Pour que l'objet soit utilisable, il est nécessaire d'appeler
   * \a setGatherGroupInfo().
   */
  GatherGroup();

  ~GatherGroup() override;

 public:

  bool needGather() override;
  void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) override;

 public:

  /*!
   * \brief Méthode permettant de définir les informations de regroupement.
   *
   * Cette méthode peut être utilisée pour remplacer les informations déjà
   * enregistrées dans l'objet.
   */
  void setGatherGroupInfo(GatherGroupInfo* ggi);

  /*!
   * \brief Méthode permettant de regrouper les données de plusieurs
   * sous-domaines sur un ou plusieurs sous-domaines.
   *
   * Il est recommandé d'utiliser cette méthode plutôt que directement
   * \a gatherToMasterIO().
   *
   * \param in Notre tableau que l'on souhaite regrouper.
   * \param out Le tableau regroupé. Si l'on n'est pas écrivain, il n'y aura
   * aucune modification.
   */
  template <class T>
  void gatherToMasterIOT(Span<const T> in, UniqueArray<T>& out);

  /*!
   * \brief Méthode permettant de regrouper les données de plusieurs
   * sous-domaines sur un ou plusieurs sous-domaines.
   *
   * Il est recommandé d'utiliser cette méthode plutôt que directement
   * \a gatherToMasterIO().
   *
   * \param in Notre tableau que l'on souhaite regrouper.
   * \param out Le tableau regroupé. Si l'on n'est pas écrivain, il n'y aura
   * aucune modification.
   */
  template <class T>
  void gatherToMasterIOT(Span2<const T> in, UniqueArray2<T>& out);

 private:

  GatherGroupInfo* m_ggi = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroupInfo::
computeSizeT(Span<const T> in)
{
  computeSize(in.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroupInfo::
computeSizeT(Span2<const T> in)
{
  computeSize(in.dim1Size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::
gatherToMasterIOT(Span<const T> in, UniqueArray<T>& out)
{
  out.clear();

  Span<const Byte> in_b(reinterpret_cast<const Byte*>(in.data()), in.sizeBytes());

  Int32 final_nb_elem = m_ggi->m_nb_elem_output;
  out.resizeNoInit(final_nb_elem);

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.data()), final_nb_elem * sizeof(T));

  gatherToMasterIO(sizeof(T), in_b, out_b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::
gatherToMasterIOT(Span2<const T> in, UniqueArray2<T>& out)
{
  out.clear();

  Span<const Byte> in_b(reinterpret_cast<const Byte*>(in.data()), in.totalNbElement() * sizeof(T));

  Int32 final_nb_elem = m_ggi->m_nb_elem_output;
  out.resizeNoInit(final_nb_elem, in.dim2Size());

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.span().data()), final_nb_elem * in.dim2Size() * sizeof(T));

  gatherToMasterIO(in.dim2Size() * sizeof(T), in_b, out_b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
