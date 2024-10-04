// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphDigest.cc                                         (C) 2000-2024 */
/*                                                                           */
/* Calcule une somme de contrôle globale des entrées/sorties Metis.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"

#include "arcane/std/internal/MetisGraphDigest.h"

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

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MetisGraphDigest::
MetisGraphDigest(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_my_rank(pm->commRank())
, m_nb_rank(pm->commSize())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  A partir de la somme locale, calcule la somme globale et retourne une chaine
 * de caractèes représentant cette somme (sur le processeur 0 seulement, les
 * autres processeurs ont une chaine vide).
 */
String MetisGraphDigest::
_digestString(ConstArrayView<Byte> my_digest)
{
  String digest_string;

  bool is_master_io = m_parallel_mng->isMasterIO();
  Int32 io_master_rank = m_parallel_mng->masterIORank();

  UniqueArray<Byte> concat_digest;

  m_parallel_mng->gatherVariable(my_digest, concat_digest, io_master_rank);

  if (is_master_io) {
    UniqueArray<Byte> final_digest;
    hash_algo.computeHash64(concat_digest, final_digest);
    digest_string = Convert::toHexaString(final_digest);
    info() << "DigestString s=" << digest_string;
  }

  return digest_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphDigest::
_computeHash(ConstArrayView<idx_t> data, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(idx_t_size * data.size(), (Byte*)data.data()), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphDigest::
_computeHash(const idx_t* data, const Integer nb, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(idx_t_size * nb, (const Byte*)data), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphDigest::
_computeHash(const real_t* data, const Integer nb, ByteArray& output)
{
  hash_algo.computeHash64(ConstArrayView<Byte>(real_t_size * nb, (const Byte*)data), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MetisGraphDigest::
computeInputDigest(const bool need_part, const int nb_options, const MetisGraphView& my_graph,
                   const idx_t* vtxdist, const idx_t* wgtflag, const idx_t* numflag, const idx_t* ncon,
                   const idx_t* nparts, const real_t* tpwgts, const real_t* ubvec, const real_t* ipc2redist,
                   const idx_t* options)
{
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

  _computeHash(vtxdist, m_nb_rank + 1, hash_value);
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

  return _digestString(hash_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MetisGraphDigest::
computeOutputDigest(const MetisGraphView& my_graph, const idx_t* edgecut)
{
  UniqueArray<Byte> hash_value;

  _computeHash(my_graph.part, hash_value);
  _computeHash(edgecut, 1, hash_value);

  return _digestString(hash_value);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
