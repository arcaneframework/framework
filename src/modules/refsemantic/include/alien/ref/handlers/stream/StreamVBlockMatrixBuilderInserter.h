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

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamVBlockMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  friend class StreamVBlockMatrixBuilderT<ValueT>;

 protected:
  BaseInserter();
  BaseInserter(StreamVBlockMatrixBuilderT<ValueT>* parent, Integer id);
  virtual ~BaseInserter();

 public:
  //!@ Méthodes visibles de l'extérieur
  void init();

  //! Identifiant de l'inserter dans son StreamVBlockMatrixBuilder
  Integer getId() const;
  //! Nombre d'itération dans l'inserter
  Integer size();
  //! Nombre de données dans l'inserter
  Integer count();
  //! Termine l'inserter (déallocation des données)
  void end();

  // FIXME: not implemented !
  //! Retourne si le filler est au début
  bool isBegin();
  //! Retourn true, le filler est à la fin
  bool isEnd();

 protected:
  void setMatrixValues(ValueT* matrix_values);

 protected:
  void _startTimer() {}
  void _stopTimer() {}

  Integer m_id;
  Integer m_index;
  Integer m_current_size;
  Integer* m_current_k;
  ValueT* m_values;
  Integer m_count;
  Integer m_size;
  UniqueArray<Integer> m_n;
  Integer m_current_block_size_row;
  Integer m_current_block_size_col;
  UniqueArray<Integer> m_block_size_row;
  UniqueArray<Integer> m_block_size_col;
  UniqueArray<Integer> m_row_index;
  UniqueArray<Integer> m_col_index;
  //! positon of entry in the Matrix CSR structure
  UniqueArray<Integer> m_data_index;
  StreamVBlockMatrixBuilderT<ValueT>* m_parent;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamVBlockMatrixBuilderT<ValueT>::Profiler
: virtual public StreamVBlockMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  Profiler() {}

 public:
  void addMatrixEntry(Integer row_index, Integer col_index);
};

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamVBlockMatrixBuilderT<ValueT>::Filler
: virtual public StreamVBlockMatrixBuilderT<ValueT>::BaseInserter
{
 protected:
  Filler() {}

 public:
  Filler& operator++();

  void addBlockData(ConstArray2View<ValueT> values);

  //! Redémarre le filler au début
  void start();

  bool isBegin();
  //! Retourn true, le filler est à la fin
  bool isEnd();
  //! Retourne l'index d'itération courante de l'inserter
  Integer index();
  //! Taille du bloc courant à insérer
  Integer currentSize();
};

// /*---------------------------------------------------------------------------*/

template <typename ValueT>
class StreamVBlockMatrixBuilderT<ValueT>::Inserter
: public StreamVBlockMatrixBuilderT<ValueT>::Profiler
, public StreamVBlockMatrixBuilderT<ValueT>::Filler
{
 private:
  friend class StreamVBlockMatrixBuilderT<ValueT>;

 public:
  Inserter(StreamVBlockMatrixBuilderT* parent, Integer id)
  : StreamVBlockMatrixBuilderT<ValueT>::BaseInserter(parent, id)
  {}

  virtual ~Inserter() {}

 private:
  // Restreint la visibilité des méthodes de Profiler
  using StreamVBlockMatrixBuilderT<ValueT>::Profiler::addMatrixEntry;

  // Restreint la visibilité des méthodes de Filler
  using StreamVBlockMatrixBuilderT<ValueT>::Filler::addBlockData;
  using StreamVBlockMatrixBuilderT<ValueT>::Filler::operator++;
  using StreamVBlockMatrixBuilderT<ValueT>::Filler::isBegin;
  using StreamVBlockMatrixBuilderT<ValueT>::Filler::isEnd;
  using StreamVBlockMatrixBuilderT<ValueT>::Filler::index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
