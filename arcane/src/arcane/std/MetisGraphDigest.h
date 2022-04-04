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
#ifndef ARCANE_STD_METISGRAPHDIGEST
#define ARCANE_STD_METISGRAPHDIGEST
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <string>
#include <mpi.h>
#include "arcane/utils/String.h"
#include "arcane/std/MetisGraph.h"

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
{
 public:
  /**
   * Calcule une somme de controle "globale" des entrees de Metis et la retourne
   * seulement pour le processeur de rang 0 dans le communicateur comm. Pour les
   * autres processeurs, la chaine retournee est vide. La somme de controle est
   * globale dans le sens ou tous les processeurs participent a sa construction.
   */
  String computeInputDigest(MPI_Comm comm, const bool need_part, const int nb_options,
                            const MetisGraphView& my_graph, const idx_t* vtxdist, const idx_t* wgtflag,
                            const idx_t* numflag, const idx_t* ncon, const idx_t* nparts,
                            const real_t* tpwgts, const real_t* ubvec, const real_t* ipc2redist,
                            const idx_t* options);

  /**
   * Calcule une somme de controle "globale" des sorties de Metis et la retourne
   * seulement pour le processeur de rang 0 dans le communicateur comm. Pour les
   * autres processeurs, la chaine retournee est vide. La somme de controle est
   * globale dans le sens ou tous les processeurs participent a sa construction.
   */
  String computeOutputDigest(MPI_Comm comm, const MetisGraphView& my_graph,
                             const idx_t* edgecut);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
