// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.cc                                                 (C) 2000-2017 */
/*                                                                           */
/* Espace de nom des types gérant le parallélisme.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/Parallel.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelTopology.h"
#include "arcane/IParallelReplication.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file Parallel.h
 *
 * \brief Fichier contenant les déclarations concernant le modèle de
 * programmation par échange de message.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* IParallelMng::
mpiCommunicator()
{
  return getMPICommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void MessagePassing::
namedBarrier(IParallelMng* pm,const String& name)
{
  // Copie les 1024 premiers caractères de name dans un tableau de Int32
  // et fait une réduction 'max' sur ce tableau. Si la réduction est différente
  // de notre valeur c'est qu'un des rang n'a pas la bonne barrière.
  ARCANE_CHECK_POINTER(pm);

  const int nsize = 256;
  Int32 sbuf[nsize];
  Int32 max_sbuf[nsize];
  Int32ArrayView buf(nsize,sbuf);
  Int32ArrayView max_buf(nsize,max_sbuf);
  ByteArrayView buf_as_bytes(nsize*sizeof(Int32),(Byte*)sbuf);
  buf.fill(0);
  Integer len = arcaneCheckArraySize(name.length());
  ByteConstArrayView name_as_bytes(len,(const Byte*)name.localstr());
  Integer name_len = math::min(buf_as_bytes.size(),name_as_bytes.size());
  buf_as_bytes.copy(name_as_bytes.subView(0,name_len));
  buf[nsize-1] = 0;
  max_buf.copy(buf);
  pm->reduce(Parallel::ReduceMax,max_buf);

  if (max_buf!=buf){
    max_buf[nsize-1] = 0;
    String max_string((const char*)max_buf.unguardedBasePointer());
    ARCANE_FATAL("Bad namedBarrier expected='{0}' found='{1}'",
                 name,max_string);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
