// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Namespace for types managing parallelism.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Parallel.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Math.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelTopology.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/SerializeBuffer.h"

#include <iostream>
#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file Parallel.h
 *
 * \brief File containing declarations concerning the message passing model.
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
  // Copies the first 1024 characters of name into an Int32 array
  // and performs a 'max' reduction on this array. If the reduction is different
  // from our value, it means that one of the ranks does not have the correct barrier.
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

  // Create a buffer to serialize the names of the variables we have
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

  // Retrieve information from other PEs.
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

  // Iterates through the list of characteristic strings and ranges into \a out_strings
  // those that are available on all ranks of \a pm
  std::set<String> common_set;
  {
    auto end_iter = string_occurences.end();
    for( Integer i=0; i<nb_string; ++i ){
      String str = input_strings[i];
      auto i_str = string_occurences.find(str);
      if (i_str==end_iter)
        // Should not happen
        continue;
      if (i_str->second!=nb_rank)
        continue;
      common_set.insert(str);
    }
  }

  // Create the final list by iterating over \a common_set
  // which is sorted alphabetically.
  common_strings.clear();
  for( const String& s : common_set )
    common_strings.add(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MessagePassing::
dumpDateAndMemoryUsage(IParallelMng* pm, ITraceMng* tm)
{
  ARCANE_CHECK_POINTER(pm);
  ARCANE_CHECK_POINTER(tm);

  Real mem_used = platform::getMemoryUsed();
  Real mem_sum = 0.0;
  Real mem_min = 0.0;
  Real mem_max = 0.0;
  Int32 mem_min_rank = 0;
  Int32 mem_max_rank = 0;
  pm->computeMinMaxSum(mem_used, mem_min, mem_max, mem_sum, mem_min_rank, mem_max_rank);
  tm->info() << "Date: " << platform::getCurrentDateTime() << " MEM=" << (Int64)(mem_used / 1e6)
             << " MAX_MEM=" << (Int64)(mem_max / 1e6)
             << " MIN_MEM=" << (Int64)(mem_min / 1e6)
             << " AVG_MEM=" << (Int64)(mem_sum / 1e6) / pm->commSize()
             << " MIN_RANK=" << mem_min_rank << " MAX_RANK=" << mem_max_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
