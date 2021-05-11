// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LZ4DeflateService.h                                         (C) 2000-2021 */
/*                                                                           */
/* Service de compression utilisant la bibliothèque 'lz4'.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IDataCompressor.h"

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

  explicit LZ4DeflateService(const ServiceBuildInfo& sbi)
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
/*!
 * \brief Service de compression utilisant la bibliothèque 'LZ4'.
 */
class LZ4DataCompressor
: public AbstractService
, public IDataCompressor
{
 public:

  explicit LZ4DataCompressor(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_name(sbi.serviceInfo()->localName())
  {
  }

 public:

  void build() override {}
  String name() const override { return m_name; }
  Int64 minCompressSize() const override { return 512; }
  void compress(Span<const std::byte> values,Array<std::byte>& compressed_values) override
  {
    // Même si supporte en théorie une taille de tableau sur 64 bits,
    // l'algorithme 'LZ4' utilise des 'int' pour les tailles et de
    // plus ne supporte pas les valeurs supérieures à LZ4_MAX_INPUT_SIZE.
    int input_size = _toInt(values.size());
    // Vérifie qu'on ne dépasse pas LZ4_MAX_INPUT_SIZE
    if (input_size>LZ4_MAX_INPUT_SIZE)
      ARCANE_THROW(IOException,"Array is too large for LZ4: size={0} max={1}",input_size,LZ4_MAX_INPUT_SIZE);

    int dest_capacity = LZ4_compressBound(input_size);
    compressed_values.resize(dest_capacity);

    char* dest = reinterpret_cast<char*>(compressed_values.data());
    const char* source = reinterpret_cast<const char*>(values.data());

    int r = LZ4_compress_default(source,dest,input_size,dest_capacity);
    if (r==0)
      ARCANE_THROW(IOException,"IO error during compression r={0}",r);
    int dest_len = r;
    if (input_size>0){
      Real ratio = (dest_len * 100.0 ) / input_size;
      info(5) << "LZ4 compress r=" << r << " source_len=" << input_size
              << " dest_len=" << dest_len << " ratio=" << ratio;
    }
    compressed_values.resize(dest_len);
  }

  void decompress(Span<const std::byte> compressed_values,Span<std::byte> values) override
  {
    char* dest = reinterpret_cast<char*>(values.data());
    int dest_len = _toInt(values.size());

    const char* source = reinterpret_cast<const char*>(compressed_values.data());
    int source_len = _toInt(compressed_values.size());

    int r = LZ4_decompress_safe(source,dest,source_len,dest_len);
    info(5) << "LZ4 decompress r=" << r << " source_len=" << source_len << " dest_len=" << dest_len;
    if (r<0)
      ARCANE_THROW(IOException,"IO error during decompression r={0}",r);
  }
 private:
  String m_name;
 private:
  int _toInt(Int64 vsize)
  {
    // Vérifie qu'on tient dans un 'int'.
    Int64 max_int_size = std::numeric_limits<int>::max();
    if (vsize>max_int_size)
      ARCANE_THROW(IOException,"Array is too large to fit in 'int' type: size={0} max={1}",vsize,max_int_size);
    return static_cast<int>(vsize);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LZ4DeflateService,
                        ServiceProperty("LZ4",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IDeflateService));

ARCANE_REGISTER_SERVICE(LZ4DataCompressor,
                        ServiceProperty("LZ4DataCompressor",ST_Application|ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IDataCompressor));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
