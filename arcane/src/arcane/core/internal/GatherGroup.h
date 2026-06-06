// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherGroup.h                                               (C) 2000-2026 */
/*                                                                           */
/* Class allowing the management of data groupings across the writer         */
/* sub-domains.                                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_GATHERGROUP_H
#define ARCANE_CORE_INTERNAL_GATHERGROUP_H

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

class GatherGroupInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the grouping of data from certain
 * sub-domains onto other sub-domains.
 *
 * The writers will be the masterIO or the masterParallelIO if
 * \a m_use_collective_io is true.
 */
class ARCANE_CORE_EXPORT GatherGroup
: public IGatherGroup
{

 public:

  /*!
   * \brief Constructor.
   * \param gather_group_info The grouping information. \a isComputed() must be
   * true.
   */
  explicit GatherGroup(GatherGroupInfo* gather_group_info);

  /*!
   * \brief Constructor.
   * For the object to be usable, it is necessary to call
   * \a setGatherGroupInfo().
   */
  GatherGroup();

  ~GatherGroup() override;

 public:

  bool isNeedGather() override;
  void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) override;

 public:

  /*!
   * \brief Method allowing the definition of grouping information.
   *
   * This method can be used to replace the information already
   * stored in the object.
   */
  void setGatherGroupInfo(GatherGroupInfo* gather_group_info);

  /*!
   * \brief Method allowing the grouping of data from several
   * sub-domains onto one or more sub-domains.
   *
   * It is recommended to use this method rather than directly
   * \a gatherToMasterIO().
   *
   * \param in Our array that we wish to group.
   * \param out The grouped array. If we are not the writer, there will
   * be no modification.
   */
  template <class T>
  void gatherToMasterIOT(Span<const T> in, Array<T>& out);

  /*!
   * \brief Method allowing the grouping of data from several
   * sub-domains onto one or more sub-domains.
   *
   * It is recommended to use this method rather than directly
   * \a gatherToMasterIO().
   *
   * \param in Our array that we wish to group.
   * \param out The grouped array. If we are not the writer, there will
   * be no modification.
   */
  template <class T>
  void gatherToMasterIOT(Span2<const T> in, Array2<T>& out);

 private:

  GatherGroupInfo* m_gather_group_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the calculation and storage of grouping information.
 *
 * The writers will be the masterIO or the masterParallelIO if
 * \a m_use_collective_io is true.
 */
class ARCANE_CORE_EXPORT GatherGroupInfo
: public IGatherGroupInfo
{
  friend class GatherGroup;

 public:

  /*!
   * \brief Constructor.
   *
   * \param parallel_mng The parallelMng containing the masterIO.
   * \param use_collective_io True if we want all processes
   * to write (with MPI-IO for example). The writers will therefore be the
   * masterParallelIO. If False, the writer will be masterIO.
   */
  GatherGroupInfo(IParallelMng* parallel_mng, bool use_collective_io);

  ~GatherGroupInfo() override;

 public:

  void computeSize(Int32 nb_elem_in) override;
  void setNeedRecompute() override { m_is_computed = false; }
  bool isComputed() override { return m_is_computed; }
  Int32 nbElemOutput() override { return m_nb_elem_output; }
  Int32 sizeOfOutput(Int32 sizeof_type) override { return m_nb_elem_output * sizeof_type; }
  SmallSpan<Int32> nbElemRecvGatherToMasterIO() override;
  Int32 nbWriterGlobal() override { return m_nb_writer_global; }

 public:

  void setCollectiveIO(bool enable) { m_use_collective_io = enable; }

  /*!
   * \brief Method allowing the calculation of grouping information.
   *
   * Collective call.
   *
   * A second call to this method will have no effect, unless \a needRecompute()
   * is called beforehand.
   *
   * \param in The view that will be shared.
   */
  template <class T>
  void computeSizeT(Span<const T> in);

  /*!
   * \brief Method allowing the calculation of grouping information.
   *
   * Collective call.
   *
   * A second call to this method will have no effect, unless \a needRecompute()
   * is called beforehand.
   *
   * \param in The view that will be shared.
   */
  template <class T>
  void computeSizeT(Span2<const T> in);

 private:

  IParallelMng* m_parallel_mng = nullptr;
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::
gatherToMasterIOT(Span<const T> in, Array<T>& out)
{
  out.clear();

  Span<const Byte> in_b(reinterpret_cast<const Byte*>(in.data()), in.sizeBytes());

  Int32 final_nb_elem = m_gather_group_info->m_nb_elem_output;
  out.resizeNoInit(final_nb_elem);

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.data()), final_nb_elem * sizeof(T));

  gatherToMasterIO(sizeof(T), in_b, out_b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::
gatherToMasterIOT(Span2<const T> in, Array2<T>& out)
{
  out.clear();

  Span<const Byte> in_b(reinterpret_cast<const Byte*>(in.data()), in.totalNbElement() * sizeof(T));

  Int32 final_nb_elem = m_gather_group_info->m_nb_elem_output;
  out.resizeNoInit(final_nb_elem, in.dim2Size());

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.span().data()), final_nb_elem * in.dim2Size() * sizeof(T));

  gatherToMasterIO(in.dim2Size() * sizeof(T), in_b, out_b);
}

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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
