// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BitonicSortT.H                                              (C) 2000-2025 */
/*                                                                           */
/* Algorithme de tri bi-tonique parallèle.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_BITONICSORT_IMPL_H
#define ARCANE_CORE_PARALLEL_BITONICSORT_IMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/parallel/BitonicSort.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Parallel
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename KeyTypeTraits> BitonicSort<KeyType, KeyTypeTraits>::
BitonicSort(IParallelMng* parallel_mng)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
{
}

template <typename KeyType, typename KeyTypeTraits> BitonicSort<KeyType, KeyTypeTraits>::
BitonicSort(IParallelMng* parallel_mng, const KeyTypeTraits& traits)
: TraceAccessor(parallel_mng->traceMng())
, m_parallel_mng(parallel_mng)
, m_traits(traits)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_init(ConstArrayView<KeyType> keys)
{
  m_init_size = keys.size();

  // Il est indispensable que tous les rangs aient la même valeur
  // de m_size
  m_size = m_parallel_mng->reduce(Parallel::ReduceMax, m_init_size);

  m_keys.resize(m_size);
  m_key_ranks.resize(m_size);
  m_key_indexes.resize(m_size);

  Int32 rank = m_parallel_mng->commRank();

  // Les valeurs de la variable aux mailles tab sont stockées
  // dans le tableau m_key
  for (Integer i = 0; i < m_init_size; ++i) {
    m_keys[i] = keys[i];
    m_key_indexes[i] = i;
    m_key_ranks[i] = rank;
  }

  // Remplit la fin du tableau avec des valeurs non significatives
  // Afin que chaque processeur ait le meme nombre d'elements
  KeyType max_value = m_traits.maxValue();
  for (Int64 i = m_init_size; i < m_size; i++) {
    m_keys[i] = max_value;
    m_key_indexes[i] = -1;
    m_key_ranks[i] = rank;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Tri bi-tonique de la variable \a m_key

  En sortie de l'appel, la variable key est entièrement classée
  par ordre croissant : les premiers éléments sont situés
  par ordre croissant sur le processeur 0, les suivants sur 
  le processeur 1, etc...
  Le rangement se fait 'en place', de sorte que la taille
  des tableaux sur les différents processeurs n'est pas modifiée.

*/
template <typename KeyType, typename KeyTypeTraits> void
BitonicSort<KeyType, KeyTypeTraits>::
sort(ConstArrayView<KeyType> keys)
{
  _init(keys);

  m_nb_merge = 0;

  info() << "BITONIC_SORT (64bit) want_rank?=" << m_want_index_and_rank
         << " size=" << m_size
         << " memsize=" << sizeof(KeyType) * m_size
         << " structsize=" << sizeof(KeyType);

  // Appel au tri local des cles
  _localHeapSort();

  // Définition du nombre de niveaux pour l'algorithme
  Int32 nb_level = 0;
  Int32 test = 1;
  Int32 nb_rank = m_parallel_mng->commSize();
  while (nb_rank > test) {
    nb_level++;
    test *= 2;
  }
  Int32 size = 1;

  // Boucle sur les niveaux de profondeurs de l'algorithme
  for (Int32 ilevel = 0; ilevel < nb_level; ++ilevel) {
    size *= 2;

    // Boucle sur les groupes de processeurs à traiter
    for (Int32 jproc = 0; jproc < nb_rank; jproc += size) {
      _mergeLevels(jproc, size);
    }
  }

  // Le nombre d'éléments valides de la liste peut-être inférieur
  // au nombre d'éléments alloués. Dès qu'une valeur est invalide,
  // on arrête la liste.
  // TODO: partir de la fin
  for (Integer i = 0; i < m_size; ++i) {
    if (!m_traits.isValid(m_keys[i])) {
      m_size = i;
      break;
    }
  }

  info() << "END_BITONIC_SORT nb_merge=" << m_nb_merge;

  m_keys.resize(m_size);
  m_key_ranks.resize(m_size);
  m_key_indexes.resize(m_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Tri bi-tonique de la variable \a key.

  En sortie de l'appel, la variable key est entièrement classée
  par ordre croissant : les premiers éléments sont situés
  par ordre croissant sur le processeur 0, les suivants sur 
  le processeur 1, etc...
  Le rangement se fait 'en place', de sorte que la taille
  des tableaux sur les différents processeurs n'est pas modifiée.
*/
template <typename KeyType, typename KeyTypeTraits> void
BitonicSort<KeyType, KeyTypeTraits>::
_mergeLevels(Int32 begin, Int32 size)
{
  //etage separateur
  for (Int32 i = 0; i < size / 2; ++i)
    _mergeProcessors(begin + i, begin + size - 1 - i);

  if (size <= 2)
    return;

  // Application de la trieuse bitonique de niveau n/2
  Int32 div_size = size;
  while (div_size > 1) {
    div_size /= 2;
    for (Int32 iproc = 0; iproc < size; iproc += div_size)
      _separator(begin + iproc, div_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Etage séparateur de l'algorithme de tri bi-tonique
 */
template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_separator(Int32 begin, Int32 size)
{
  for (Int32 i = 0; i < size / 2; ++i)
    _mergeProcessors(begin + i, begin + i + size / 2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fusion des listes de deux processeurs.
 *
 \param iproc1 processeur de plus faible Id dont on fusionne la liste
 \param iproc2 processeur de plus fort Id dont on fusionne la liste

 * Cette opération est la brique élémentaire de 
 * l'algorithme de tri fusion bi-tonique. En entrée, elle
 * prend les numéros de deux processeurs avec iproc1<iproc2.
 * En sortie, les  tableaux m_key de ces deux processeurs
 * sont triés : les plus petits éléments sont triés par ordre croissant
 * sur iproc1, les plus grands éléments sont triés par ordre croissant
 * sur iproc2.
 */
template <typename KeyType, typename KeyTypeTraits> void BitonicSort<KeyType, KeyTypeTraits>::
_mergeProcessors(Int32 iproc1, Int32 iproc2)
{
  if (iproc1 >= iproc2)
    ARCANE_FATAL("Invalid merge iproc1={0} iproc2={1}", iproc1, iproc2);

  Int32 my_rank = m_parallel_mng->commRank();

  if (iproc2 >= m_parallel_mng->commSize())
    return;

  bool is_proc2 = (iproc2 == my_rank);
  bool is_proc1 = (iproc1 == my_rank);

  if (!is_proc1 && !is_proc2)
    return;

  info() << "SORT iproc1=" << iproc1 << " iproc2=" << iproc2;
  ++m_nb_merge;

  const Int64 buf_size = m_size;
  const bool is_want_index_rank = m_want_index_and_rank;
  ////////////////////////////////////////////
  //traitement du processeur 2
  ////////////////////////////////////////////

  if (is_proc2) {
    //envoi de la liste au proc 1
    UniqueArray<KeyType> send_keys(m_keys);
    UniqueArray<Request> requests;
    requests.add(m_traits.send(m_parallel_mng, iproc1, send_keys));
    if (is_want_index_rank) {
      requests.add(m_parallel_mng->send(m_key_indexes, iproc1, false));
      requests.add(m_parallel_mng->send(m_key_ranks, iproc1, false));
    }
    //reception de la fin de la liste fusionnee par le proc1
    requests.add(m_traits.recv(m_parallel_mng, iproc1, m_keys));
    if (is_want_index_rank) {
      requests.add(m_parallel_mng->recv(m_key_indexes, iproc1, false));
      requests.add(m_parallel_mng->recv(m_key_ranks, iproc1, false));
    }
    m_parallel_mng->waitAllRequests(requests);
  }

  //////////////////////////////////////
  //traitement du processeur 1
  //////////////////////////////////////
  if (is_proc1) {
    //reception de la liste du proc2

    UniqueArray<KeyType> buf2(buf_size);
    Int32UniqueArray buf2_proc;
    Int32UniqueArray buf2_local_id;
    //on alloue buf2 a la taille+1 afin de placer une sentinelle en
    //buf2[size]
    buf2.resize(buf_size);
    if (is_want_index_rank) {
      buf2_proc.resize(buf_size);
      buf2_local_id.resize(buf_size);
    }

    Request recv_request = m_traits.recv(m_parallel_mng, iproc2, buf2);
    m_parallel_mng->waitAllRequests(ArrayView<Request>(1, &recv_request));
    if (is_want_index_rank) {
      m_parallel_mng->recv(buf2_local_id, iproc2);
      m_parallel_mng->recv(buf2_proc, iproc2);
    }

    //à remplacer par le plus grand nombre representable par un Integer
    buf2.add(m_traits.maxValue());

    // allocation d'un buffer de travail pour la fusion des deux listes
    Int64 total_size = m_size + buf_size;
    UniqueArray<KeyType> buf_merge(total_size);
    UniqueArray<Int32> buf_proc_merge;
    UniqueArray<Int32> buf_local_id_merge;
    if (is_want_index_rank) {
      buf_proc_merge.resize(total_size);
      buf_local_id_merge.resize(total_size);
    }

#if 0
    info() << "SIZE=" << buf_size << " total=" << total_size << " m_size=" << m_size
           << " proc2=" << iproc2 << " init-size=" << m_init_size;
#endif

    // Fusion des deux listes en partant du principe que chacune est
    // déjà triée.
    Int64 cursor1 = 0;
    Int64 cursor2 = 0;
    for (Int64 i = 0; i < total_size; ++i) {
      if (cursor1 >= m_size || (m_traits.compareLess(buf2[cursor2], m_keys[cursor1]) && cursor2 < buf_size)) {
        buf_merge[i] = buf2[cursor2];
        if (is_want_index_rank) {
          buf_local_id_merge[i] = buf2_local_id[cursor2];
          buf_proc_merge[i] = buf2_proc[cursor2];
        }
        ++cursor2;
      }
      else {
        buf_merge[i] = m_keys[cursor1];
        if (is_want_index_rank) {
          buf_local_id_merge[i] = m_key_indexes[cursor1];
          buf_proc_merge[i] = m_key_ranks[cursor1];
        }
        ++cursor1;
      }
    }
    buf2.resize(buf_size);

    // Recopie et envoi de la fin de la liste vers le proc2
    for (Integer ii = 0; ii < buf_size; ii++) {
      buf2[ii] = buf_merge[m_size + ii];
      if (is_want_index_rank) {
        buf2_local_id[ii] = buf_local_id_merge[m_size + ii];
        buf2_proc[ii] = buf_proc_merge[m_size + ii];
      }
    }
    Request send_request = m_traits.send(m_parallel_mng, iproc2, buf2);
    m_parallel_mng->waitAllRequests(ArrayView<Request>(1, &send_request));
    if (is_want_index_rank) {
      m_parallel_mng->send(buf2_local_id, iproc2);
      m_parallel_mng->send(buf2_proc, iproc2);
    }

    // recopie du debut de la liste dans la variable m_work_unique_id
    for (Integer i = 0; i < buf_size; i++) {
      m_keys[i] = buf_merge[i];
      if (is_want_index_rank) {
        m_key_indexes[i] = buf_local_id_merge[i];
        m_key_ranks[i] = buf_proc_merge[i];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tri par tas en local de la variable \a m_key.
 *
 * En sortie de l'appel, la variable est classée
 * par ordre croissant à l'intérieur de chaque processeur.
 */
template <typename KeyType, typename KeyTypeTraits>
void BitonicSort<KeyType, KeyTypeTraits>::
_localHeapSort()
{
  // Cas particulier d'une liste réduite à zero ou un element
  if (m_size < 2)
    return;

  // Suivant du parent du dernier element
  Int64 l = m_size / 2;
  Int64 ir = m_size - 1;
  KeyType rra;
  Int64 i = 0;
  Int64 j = 0;
  Integer tmp_local_id = NULL_ITEM_LOCAL_ID;
  Integer tmp_proc = A_NULL_RANK;

  for (;;) {
    if (l > 0) {
      --l;
      rra = m_keys[l];
      tmp_local_id = m_key_indexes[l];
      tmp_proc = m_key_ranks[l];
    }
    else {
      rra = m_keys[ir];
      tmp_local_id = m_key_indexes[ir];
      tmp_proc = m_key_ranks[ir];

      m_keys[ir] = m_keys[0];
      m_key_indexes[ir] = m_key_indexes[0];
      m_key_ranks[ir] = m_key_ranks[0];

      if (--ir == 0) {
        m_keys[0] = rra;
        m_key_indexes[0] = tmp_local_id;
        m_key_ranks[0] = tmp_proc;
        break;
      }
    }
    i = l;
    j = l + l + 1;
    while (j <= ir) {
      if (j < ir && m_traits.compareLess(m_keys[j], m_keys[j + 1]))
        ++j;
      if (m_traits.compareLess(rra, m_keys[j])) {
        m_keys[i] = m_keys[j];
        m_key_indexes[i] = m_key_indexes[j];
        m_key_ranks[i] = m_key_ranks[j];
        i = j;
        //enfant a gauche
        j = j * 2 + 1;
      }
      else
        j = ir + 1;
    }
    m_keys[i] = rra;
    m_key_indexes[i] = tmp_local_id;
    m_key_ranks[i] = tmp_proc;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Parallel

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
