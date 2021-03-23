// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Bzip2DeflateService.h                                       (C) 2000-2020 */
/*                                                                           */
/* Service de compression utilisant la bibliothèque 'bzip2'.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"
#include "arcane/IDeflateService.h"

#include <bzlib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de compression utilisant la bibliothèque 'Bzip2'.
 */
class Bzip2DeflateService
: public AbstractService
, public IDeflateService
{
 public:
  Bzip2DeflateService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
  }

 public:

  void build() override {}

  void compress(ByteConstArrayView values,ByteArray& compressed_values) override
  {
    Integer input_size = values.size();
    // D'après la doc, il faut allouer au moins 1% de plus que la taille
    // d'entrée plus encore 600 bytes
    Integer compressed_init_size = (Integer)(((Real)input_size) * 1.01) + 600;
    compressed_values.resize(compressed_init_size);
    //compressed_values.copy(values);

    char* dest = (char*)compressed_values.data();
    unsigned int dest_len = (unsigned int)compressed_init_size;

    char* source = (char*)values.data();
    unsigned int source_len = (unsigned int)input_size;
    
    int blockSize100k = 9;
    int verbosity = 1;
    int workFactor = 30;
    
#if 0
    info() << "CHECK COMPRESS dest=" << (void*)dest
           << " dest_len=" << dest_len
           << " source=" << (void*)source
           << " source_len=" << source_len;
#endif
    
    int r = BZ2_bzBuffToBuffCompress(dest, 
                                     &dest_len,
                                     source, 
                                     source_len,
                                     blockSize100k, 
                                     verbosity, 
                                     workFactor);
    if (r!=BZ_OK)
      throw IOException(A_FUNCINFO,String::format("io error during compression r={0}",r));
    // Attention overflow des Int32;
    Real ratio = (dest_len * 100 ) / source_len;
    info() << "Bzip2 compress r=" << r << " source_len=" << source_len
           << " dest_len=" << dest_len << " ratio=" << ratio;
    compressed_values.resize(dest_len);
  }

  void decompress(ByteConstArrayView compressed_values,ByteArrayView values) override
  {
    char* dest = (char*)values.data();
    unsigned int dest_len = (unsigned int)values.size();

    char* source = (char*)compressed_values.data();
    unsigned int source_len = (unsigned int)compressed_values.size();

    // small vaut 1 si on souhaite economiser la memoire (mais c'est moins rapide)
    // et 0 sinon.
    int small = 0;
    int verbosity = 1;
    int r = BZ2_bzBuffToBuffDecompress(dest,&dest_len,
                                       source,source_len,
                                       small,verbosity);
    info() << "Bzip2 decompress r=" << r << " source_len=" << source_len
           << " dest_len=" << dest_len;
    if (r!=BZ_OK)
      throw IOException(A_FUNCINFO,String::format("io error during decompression r={0}",r));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(Bzip2DeflateService,
                                    IDeflateService,
                                    Bzip2);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
