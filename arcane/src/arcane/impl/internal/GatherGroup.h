// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GatherGroup.h                                            (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire de fabriques de maillages.                                   */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT GatherGroup
: public IGatherGroup
{
 public:

  class ARCANE_IMPL_EXPORT GatherGroupInfo
  : public IGatherGroupInfo
  {
    friend GatherGroup;

   public:

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

    template <class T>
    void computeSizeT(Span<const T> in);

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

 public:

  explicit GatherGroup(GatherGroupInfo* ggi);
  GatherGroup();
  ~GatherGroup() override;

 public:

  bool needGather() override;
  void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) override;

 public:

  void setGatherGroupInfo(GatherGroupInfo* ggi);

  template <class T>
  void gatherToMasterIOT(Span<const T> in, UniqueArray<T>& out);

  template <class T>
  void gatherToMasterIOT(Span2<const T> in, UniqueArray<T>& out);

 private:

  GatherGroupInfo* m_ggi = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::GatherGroupInfo::
computeSizeT(Span<const T> in)
{
  computeSize(in.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::GatherGroupInfo::
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
  out.resize(final_nb_elem);

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.data()), final_nb_elem * sizeof(T));

  gatherToMasterIO(sizeof(T), in_b, out_b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void GatherGroup::
gatherToMasterIOT(Span2<const T> in, UniqueArray<T>& out)
{
  out.clear();

  Span<const Byte> in_b(reinterpret_cast<const Byte*>(in.data()), in.totalNbElement() * sizeof(T));

  Int32 final_nb_elem = m_ggi->m_nb_elem_output;
  out.resize(final_nb_elem * in.dim2Size());

  Span<Byte> out_b(reinterpret_cast<Byte*>(out.data()), final_nb_elem * in.dim2Size() * sizeof(T));

  gatherToMasterIO(in.dim2Size() * sizeof(T), in_b, out_b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
