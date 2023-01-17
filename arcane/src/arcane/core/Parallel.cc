// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.cc                                                 (C) 2000-2022 */
/*                                                                           */
/* Espace de nom des types gérant le parallélisme.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Parallel.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IParallelMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelTopology.h"
#include "arcane/IParallelReplication.h"
#include "arcane/SerializeBuffer.h"

#include <iostream>
#include <map>
#include <set>

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

namespace Arcane
{

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

extern "C++" void MessagePassing::
filterCommonStrings(IParallelMng* pm,ConstArrayView<String> input_strings,
                    Array<String>& common_strings)
{
  const Int32 nb_string = input_strings.size();

  // Créé un buffer pour sérialiser les noms des variables dont on dispose
  SerializeBuffer send_buf;
  send_buf.setMode(ISerializer::ModeReserve);
  send_buf.reserve(DT_Int32,1);
  for( Integer i=0; i<nb_string; ++i ){
    send_buf.reserve(input_strings[i]);
  }

  send_buf.allocateBuffer();
  send_buf.setMode(ISerializer::ModePut);
  send_buf.putInt32(nb_string);
  for( Integer i=0; i<nb_string; ++i ){
    send_buf.put(input_strings[i]);
  }

  // Récupère les infos des autres PE.
  SerializeBuffer recv_buf;
  pm->allGather(&send_buf,&recv_buf);

  std::map<String,Int32> string_occurences;

  Int32 nb_rank = pm->commSize();
  recv_buf.setMode(ISerializer::ModeGet);
  for( Integer i=0; i<nb_rank; ++i ){
    Int32 nb_string_rank = recv_buf.getInt32();
    for( Integer z=0; z<nb_string_rank; ++z ){
      String x;
      recv_buf.get(x);
      auto vo = string_occurences.find(x);
      if (vo==string_occurences.end())
        string_occurences.insert(std::make_pair(x,1));
      else
        vo->second = vo->second + 1;
    }
  }

  // Parcours la liste des chaînes de caractète et range dans \a out_strings
  // celles qui sont disponibles sur tous les rangs de \a pm
  std::set<String> common_set;
  {
    auto end_iter = string_occurences.end();
    for( Integer i=0; i<nb_string; ++i ){
      String str = input_strings[i];
      auto i_str = string_occurences.find(str);
      if (i_str==end_iter)
        // Ne devrait pas arriver
        continue;
      if (i_str->second!=nb_rank)
        continue;
      common_set.insert(str);
    }
  }

  // Créé la liste finale en itérant sur \a common_set
  // qui est trié alphabétiquement.
  common_strings.clear();
  for( const String& s : common_set )
    common_strings.add(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
