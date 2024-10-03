// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MetisGraphDigest.h                                          (C) 2000-2024 */
/*                                                                           */
/* Calcule une somme de contrôle globale des entrées/sorties Metis.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_METISGRAPHDIGEST
#define ARCANE_STD_METISGRAPHDIGEST
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

#include "arcane/std/MetisGraph.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule une somme de contrôle globale des entrées/sorties Metis.
 */
class MetisGraphDigest
: public TraceAccessor
{
 public:

  explicit MetisGraphDigest(IParallelMng* pm);

 public:

  /*!
   * Calcule une somme de controle "globale" des entrees de Metis et la retourne
   * seulement pour le processeur de rang 0 dans le communicateur comm. Pour les
   * autres processeurs, la chaine retournee est vide. La somme de controle est
   * globale dans le sens ou tous les processeurs participent a sa construction.
   */
  String computeInputDigest(const bool need_part, const int nb_options,
                            const MetisGraphView& my_graph, const idx_t* vtxdist, const idx_t* wgtflag,
                            const idx_t* numflag, const idx_t* ncon, const idx_t* nparts,
                            const real_t* tpwgts, const real_t* ubvec, const real_t* ipc2redist,
                            const idx_t* options);

  /*!
   * Calcule une somme de controle "globale" des sorties de Metis et la retourne
   * seulement pour le processeur de rang 0 dans le communicateur comm. Pour les
   * autres processeurs, la chaine retournee est vide. La somme de controle est
   * globale dans le sens ou tous les processeurs participent a sa construction.
   */
  String computeOutputDigest(const MetisGraphView& my_graph, const idx_t* edgecut);

 private:

  IParallelMng* m_parallel_mng = nullptr;
  Int32 m_my_rank = A_NULL_RANK;
  Int32 m_nb_rank = 0;

 private:

  void _computeHash(ConstArrayView<idx_t> data, ByteArray& output);
  void _computeHash(const idx_t* data, const Integer nb, ByteArray& output);
  void _computeHash(const real_t* data, const Integer nb, ByteArray& output);
  String _digestString(ConstArrayView<Byte> my_digest);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
