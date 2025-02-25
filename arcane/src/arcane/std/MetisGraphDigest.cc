// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphDigest.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Calcule une somme de contrôle globale des entrées/sorties Metis.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

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
 * de caractères représentant cette somme (sur le processeur 0 seulement, les
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
_computeHash(Span<const idx_t> data, ByteArray& output, const char* name)
{
  UniqueArray<Byte> bytes;
  hash_algo.computeHash64(Arccore::asBytes(data), bytes);
  info() << "COMPUTE_HASH=" << name << " v=" << Convert::toHexaString(bytes);
  output.addRange(bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MetisGraphDigest::
_computeHash(Span<const real_t> data, ByteArray& output, const char* name)
{
  UniqueArray<Byte> bytes;
  hash_algo.computeHash64(Arccore::asBytes(data), bytes);
  info() << "COMPUTE_HASH=" << name << " v=" << Convert::toHexaString(bytes);
  output.addRange(bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#define COMPUTE_HASH1(array, output) _computeHash(array, output, #array)
#define COMPUTE_HASH(array, n, output) _computeHash({ array, n }, output, #array)

String MetisGraphDigest::
computeInputDigest(const bool need_part, const int nb_options, const MetisGraphView& my_graph,
                   const idx_t* vtxdist, const idx_t* wgtflag, const idx_t* numflag, const idx_t* ncon,
                   const idx_t* nparts, const real_t* tpwgts, const real_t* ubvec, const real_t* ipc2redist,
                   const idx_t* options)
{
  UniqueArray<Byte> hash_value;

  // Signature du graph lui-meme

  COMPUTE_HASH1(my_graph.xadj, hash_value);
  COMPUTE_HASH1(my_graph.adjncy, hash_value);
  COMPUTE_HASH1(my_graph.vwgt, hash_value);

  if (my_graph.have_vsize) {
    COMPUTE_HASH1(my_graph.vsize, hash_value);
  }

  if (my_graph.have_adjwgt) {
    COMPUTE_HASH1(my_graph.adjwgt, hash_value);
  }

  if (need_part) {
    COMPUTE_HASH1(my_graph.part, hash_value);
  }

  // Ajout de la signature des options, des dimensions

  COMPUTE_HASH(vtxdist, m_nb_rank + 1, hash_value);
  COMPUTE_HASH(wgtflag, 1, hash_value);
  COMPUTE_HASH(numflag, 1, hash_value);
  COMPUTE_HASH(ncon, 1, hash_value);
  COMPUTE_HASH(nparts, 1, hash_value);
  COMPUTE_HASH(tpwgts, (*nparts) * (*ncon), hash_value);
  COMPUTE_HASH(ubvec, (*ncon), hash_value);

  if (ipc2redist) {
    COMPUTE_HASH(ipc2redist, 1, hash_value);
  }

  // Cf. doc Metis : si options[0] == 1 alors la taille de "options" est 3 ou 4
  if ((*options) == 1) {
    COMPUTE_HASH(options, nb_options, hash_value);
  }
  else {
    COMPUTE_HASH(options, 1, hash_value);
  }

  return _digestString(hash_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MetisGraphDigest::
computeOutputDigest(const MetisGraphView& my_graph, const idx_t* edgecut)
{
  UniqueArray<Byte> hash_value;

  COMPUTE_HASH1(my_graph.part, hash_value);
  COMPUTE_HASH(edgecut, 1, hash_value);

  return _digestString(hash_value);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
