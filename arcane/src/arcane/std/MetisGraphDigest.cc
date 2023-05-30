// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphDigest.h                                          (C) 2000-2019 */
/*                                                                           */
/* Calcule une somme de contrôle globale des entrées/sorties Metis.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/std/MetisGraphDigest.h"

#include <vector>
#include <numeric>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
MD5HashAlgorithm hash_algo;
const Integer idx_t_size = sizeof(idx_t);
const Integer real_t_size = sizeof(real_t);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_computeHash(ConstArrayView<idx_t> data, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(idx_t_size * data.size(), (Byte*)data.data()), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_computeHash(const idx_t* data, const Integer nb, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(idx_t_size * nb, (const Byte*)data), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_computeHash(const real_t* data, const Integer nb, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(real_t_size * nb, (const Byte*)data), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  A partir de la somme locale, calcule la somme globale et retourne une chaine
 * de caracteres representant cette somme (sur le processeur 0 seulement, les
 * autres processeurs ont une chaine vide).
 */
String
_digestString(MPI_Comm comm, const int my_rank, const int nb_rank, const int io_rank,
              ByteArray& my_digest)
{
  UniqueArray<Byte> concat_digest;

  int my_digest_size = my_digest.size();

  std::vector<int> digest_offset(nb_rank);
  std::vector<int> digest_size(nb_rank);

  MPI_Gather(&my_digest_size, 1, MPI_INT, digest_size.data(), 1, MPI_INT, io_rank, comm);

  if (my_rank == io_rank) {
    int concat_digest_size = std::accumulate(digest_size.begin(), digest_size.end(), 0);
    concat_digest.resize(concat_digest_size);
    digest_offset[0] = 0;
    for (int i = 1; i < nb_rank; ++i) {
      digest_offset[i] = digest_offset[i - 1] + digest_size[i - 1];
    }
  }

  MPI_Gatherv(my_digest.data(), my_digest_size, MPI_BYTE, concat_digest.data(),
              digest_size.data(), digest_offset.data(), MPI_BYTE, io_rank, comm);

  if (my_rank == io_rank) {
    UniqueArray<Byte> final_digest;
    hash_algo.computeHash64(concat_digest, final_digest);
    return Convert::toHexaString(final_digest);
  }

  return String();
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MetisGraphDigest::
computeInputDigest(MPI_Comm comm, const bool need_part, const int nb_options, const MetisGraphView& my_graph,
                   const idx_t* vtxdist, const idx_t* wgtflag, const idx_t* numflag, const idx_t* ncon,
                   const idx_t* nparts, const real_t* tpwgts, const real_t* ubvec, const real_t* ipc2redist,
                   const idx_t* options)
{
  int nb_rank = -1;
  int my_rank = -1;
  int io_rank = 0;

  MPI_Comm_size(comm, &nb_rank);
  MPI_Comm_rank(comm, &my_rank);

  UniqueArray<Byte> hash_value;

  // Signature du graph lui-meme

  _computeHash(my_graph.xadj, hash_value);
  _computeHash(my_graph.adjncy, hash_value);
  _computeHash(my_graph.vwgt, hash_value);

  if (my_graph.have_vsize) {
    _computeHash(my_graph.vsize, hash_value);
  }

  if (my_graph.have_adjwgt) {
    _computeHash(my_graph.adjwgt, hash_value);
  }

  if (need_part) {
    _computeHash(my_graph.part, hash_value);
  }

  // Ajout de la signature des options, des dimensions

  _computeHash(vtxdist, nb_rank + 1, hash_value);
  _computeHash(wgtflag, 1, hash_value);
  _computeHash(numflag, 1, hash_value);
  _computeHash(ncon, 1, hash_value);
  _computeHash(nparts, 1, hash_value);
  _computeHash(tpwgts, (*nparts) * (*ncon), hash_value);
  _computeHash(ubvec, (*ncon), hash_value);

  if (ipc2redist) {
    _computeHash(ipc2redist, 1, hash_value);
  }

  // Cf. doc Metis : si options[0] == 1 alors la taille de "options" est 3 ou 4
  if ((*options) == 1) {
    _computeHash(options, nb_options, hash_value);
  }
  else {
    _computeHash(options, 1, hash_value);
  }

  return _digestString(comm, my_rank, nb_rank, io_rank, hash_value);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MetisGraphDigest::
computeOutputDigest(MPI_Comm comm, const MetisGraphView& my_graph, const idx_t* edgecut)
{
  int nb_rank = -1;
  int my_rank = -1;
  int io_rank = 0;

  MPI_Comm_size(comm, &nb_rank);
  MPI_Comm_rank(comm, &my_rank);

  UniqueArray<Byte> hash_value;

  _computeHash(my_graph.part, hash_value);
  _computeHash(edgecut, 1, hash_value);

  return _digestString(comm, my_rank, nb_rank, io_rank, hash_value);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
