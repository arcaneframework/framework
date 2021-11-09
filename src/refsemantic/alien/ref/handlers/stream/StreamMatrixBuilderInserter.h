/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <alien/ref/AlienRefSemanticPrecomp.h>
/*---------------------------------------------------------------------------*/

namespace Alien
{
using namespace Arccore;

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  friend class StreamMatrixBuilderT<ValueT>;

 protected:
  BaseInserter();
  BaseInserter(StreamMatrixBuilderT<ValueT>* parent, Integer id);
  virtual ~BaseInserter();

 public:
  //!@ Méthodes visibles de l'extérieur
  void init();

  //! Identifiant de l'inserter dans son StreamMatrixBuilder
  Integer getId() const;
  //! Nombre d'itération dans l'inserter
  Integer size();
  //! Nombre de données dans l'inserter
  Integer count();
  //! Termine l'inserter (déallocation des données)
  void end();

 protected:
  void setMatrixValues(ValueT* matrix_values, Integer block_size = 1);

 protected:
  void _startTimer() {}
  void _stopTimer() {}

  Integer m_id;
  Integer m_index;
  Integer m_current_size;
  Integer* m_current_k;
#if defined(WIN32) or defined(__clang__)
  ValueT* __restrict m_values;
#else
  ValueT __restrict__* m_values;
#endif
  Integer m_count;
  Integer m_size;
  Integer m_block_size;
  UniqueArray<Integer> m_n;
  UniqueArray<Integer> m_row_index;
  UniqueArray<Integer> m_col_index;
  //! positon of entry in the Matrix CSR structure
  UniqueArray<Integer> m_data_index;
  StreamMatrixBuilderT<ValueT>* m_parent;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamMatrixBuilderT<ValueT>::Profiler
: virtual public StreamMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  Profiler() {}

 public:
  void reserve(Integer capacity);
  void addMatrixEntries(
  ConstArrayView<Integer> row_index, ConstArrayView<Integer> col_index);
  void addMatrixEntries(ConstArrayView<Integer> row_indexes,
                        const UniqueArray<ConstArrayView<Integer>>& col_indexes);
  void addMatrixEntries(ConstArrayView<Integer> row_indexes,
                        UniqueArray2<Integer> col_indexes, ConstArrayView<Integer> stencil_lids,
                        Integer size);
  void addMatrixEntries(const Integer row_index, ConstArrayView<Integer> col_index);
  void addMatrixEntry(Integer row_index, Integer col_index);
};

// /*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamMatrixBuilderT<ValueT>::Filler
: virtual public StreamMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  Filler() {}

 public:
  void setData(ConstArrayView<ValueT> values);
  void setData(ValueT values);
  void addData(ConstArrayView<ValueT> values);
  void addData(ConstArrayView<ValueT> values, ValueT factor);
  void addData(ValueT values);
  void addMultiData(ValueT value);
  void addBlockData(ConstArrayView<ValueT> values);
  void addBlockData(ConstArrayView<ValueT> values, ValueT factor);
  void addMultiData(ConstArrayView<ValueT> values, Integer size);
  void addMultiData(ConstArrayView<ValueT> values, ValueT factor, Integer size);
  Filler& operator++();

  //! Redémarre le filler au début
  void start();
  //! Positionne le filler est à la fin
  void setEnd();
  //! Taille du bloc courant à insérer
  Integer currentSize();
  //! Retourne true si le filler est au début
  bool isBegin();
  //! Retourn true, le filler est à la fin
  bool isEnd();
  //! Retourne l'index d'itération courante de l'inserter
  Integer index();
};

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamMatrixBuilderT<ValueT>::Inserter
: public StreamMatrixBuilderT<ValueT>::Profiler
, public StreamMatrixBuilderT<ValueT>::Filler
{
 private:
  friend class StreamMatrixBuilderT<ValueT>;

 public:
  Inserter(StreamMatrixBuilderT* parent, Integer id)
  : StreamMatrixBuilderT<ValueT>::BaseInserter(parent, id)
  {}

  virtual ~Inserter() {}

 private:
  // Restreint la visibilité des méthodes de Profiler
  using StreamMatrixBuilderT<ValueT>::Profiler::reserve;
  using StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntry;
  using StreamMatrixBuilderT<ValueT>::Profiler::addMatrixEntries;

  // Restreint la visibilité des méthodes de Filler
  using StreamMatrixBuilderT<ValueT>::Filler::setData;
  using StreamMatrixBuilderT<ValueT>::Filler::addData;
  using StreamMatrixBuilderT<ValueT>::Filler::addMultiData;
  using StreamMatrixBuilderT<ValueT>::Filler::addBlockData;
  using StreamMatrixBuilderT<ValueT>::Filler::operator++;
  using StreamMatrixBuilderT<ValueT>::Filler::start;
  using StreamMatrixBuilderT<ValueT>::Filler::setEnd;
  using StreamMatrixBuilderT<ValueT>::Filler::currentSize;
  using StreamMatrixBuilderT<ValueT>::Filler::isBegin;
  using StreamMatrixBuilderT<ValueT>::Filler::isEnd;
  using StreamMatrixBuilderT<ValueT>::Filler::index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
