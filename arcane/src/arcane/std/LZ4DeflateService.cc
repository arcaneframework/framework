// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LZ4DeflateService.h                                         (C) 2000-2020 */
/*                                                                           */
/* Service de compression utilisant la bibliothèque 'lz4'.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"
#include "arcane/IDeflateService.h"

#include <lz4.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de compression utilisant la bibliothèque 'LZ4'.
 */
class LZ4DeflateService
: public AbstractService
, public IDeflateService
{
 public:
  LZ4DeflateService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
  }

 public:

  void build() override {}

  void compress(ByteConstArrayView values,ByteArray& compressed_values) override
  {
    Integer input_size = values.size();
    int dest_capacity = LZ4_compressBound(input_size);
    // D'après la doc, il faut allouer au moins 1% de plus que la taille
    // d'entrée plus encore 600 bytes
    //Integer compressed_init_size = (Integer)(((Real)input_size) * 1.01) + 600;
    compressed_values.resize(dest_capacity);
    //compressed_values.copy(values);

    char* dest = reinterpret_cast<char*>(compressed_values.data());

    const char* source = reinterpret_cast<const char*>(values.data());
    unsigned int source_len = (unsigned int)input_size;
    
#if 0
    info() << "CHECK COMPRESS dest=" << (void*)dest
           << " dest_len=" << dest_len
           << " source=" << (void*)source
           << " source_len=" << source_len;
#endif
    
    int r = LZ4_compress_default(source,dest,source_len,dest_capacity);

    if (r==0)
      throw IOException(A_FUNCINFO,String::format("io error during compression r={0}",r));
    int dest_len = r;
    Real ratio = (dest_len * 100.0 ) / source_len;
    info() << "LZ4 compress r=" << r << " source_len=" << source_len
           << " dest_len=" << dest_len << " ratio=" << ratio;
    compressed_values.resize(dest_len);
  }

  void decompress(ByteConstArrayView compressed_values,ByteArrayView values) override
  {
    char* dest = reinterpret_cast<char*>(values.data());
    int dest_len = values.size();

    const char* source = reinterpret_cast<const char*>(compressed_values.data());
    int source_len = compressed_values.size();

    int r = LZ4_decompress_safe(source,dest,source_len,dest_len);
    info() << "LZ4 decompress r=" << r << " source_len=" << source_len << " dest_len=" << dest_len;
    if (r<0)
      throw IOException(A_FUNCINFO,String::format("io error during decompression r={0}",r));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(LZ4DeflateService,
                                    IDeflateService,
                                    LZ4);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
