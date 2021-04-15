// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Bzip2DeflateService.h                                       (C) 2000-2021 */
/*                                                                           */
/* Service de compression utilisant la bibliothèque 'bzip2'.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IDataCompressor.h"

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
      ARCANE_THROW(IOException,"IO error during decompression r={0}",r);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de compression utilisant la bibliothèque 'Bzip2'.
 */
class Bzip2DataCompressor
: public AbstractService
, public IDataCompressor
{
 public:
  explicit Bzip2DataCompressor(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_name(sbi.serviceInfo()->localName())
  {
  }

 public:

  void build() override {}
  String name() const override { return m_name; }
  void compress(Span<const std::byte> values,Array<std::byte>& compressed_values) override
  {
    Int64 input_size = values.size();
    // D'après la doc, il faut allouer au moins 1% de plus que la taille
    // d'entrée plus encore 600 bytes
    Integer compressed_init_size = (Integer)(((Real)input_size) * 1.01) + 600;
    compressed_values.resize(compressed_init_size);

    char* dest = reinterpret_cast<char*>(compressed_values.data());
    unsigned int dest_len = _toUInt(compressed_init_size);

    char* source = const_cast<char*>(reinterpret_cast<const char*>(values.data()));
    unsigned int source_len = _toUInt(input_size);

    int blockSize100k = 9;
    int verbosity = 1;
    int workFactor = 30;

    int r = BZ2_bzBuffToBuffCompress(dest,
                                     &dest_len,
                                     source,
                                     source_len,
                                     blockSize100k,
                                     verbosity,
                                     workFactor);
    if (r!=BZ_OK)
      ARCANE_THROW(IOException,"IO error during compression r={0}",r);

    // Attention overflow des Int32;
    Real ratio = (dest_len * 100 ) / source_len;
    info() << "Bzip2 compress r=" << r << " source_len=" << source_len
           << " dest_len=" << dest_len << " ratio=" << ratio;
    compressed_values.resize(dest_len);
  }

  void decompress(Span<const std::byte> compressed_values,Span<std::byte> values) override
  {
    char* dest = reinterpret_cast<char*>(values.data());
    unsigned int dest_len = _toUInt(values.size());

    char* source = const_cast<char*>(reinterpret_cast<const char*>(compressed_values.data()));
    unsigned int source_len = _toUInt(compressed_values.size());

    // small vaut 1 si on souhaite economiser la memoire (mais c'est moins rapide)
    // et 0 sinon.
    int small = 0;
    int verbosity = 1;
    int r = BZ2_bzBuffToBuffDecompress(dest,&dest_len,
                                       source,source_len,
                                       small,verbosity);
    info(5) << "Bzip2 decompress r=" << r << " source_len=" << source_len
           << " dest_len=" << dest_len;
    if (r!=BZ_OK)
      ARCANE_THROW(IOException,"IO error during decompression r={0}",r);
  }
 private:
  String m_name;
 private:
  unsigned int _toUInt(Int64 vsize)
  {
    // Vérifie qu'on tient dans un 'int'.
    Int64 max_uint_size = std::numeric_limits<unsigned int>::max();
    if (vsize>max_uint_size)
      ARCANE_THROW(IOException,"Array is too large to fit in 'unsigned int' type: size={0} max={1}",
                   vsize,max_uint_size);
    return static_cast<unsigned int>(vsize);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(Bzip2DeflateService,
                        ServiceProperty("Bzip2",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IDeflateService));

ARCANE_REGISTER_SERVICE(Bzip2DataCompressor,
                        ServiceProperty("Bzip2DataCompressor",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IDataCompressor));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
