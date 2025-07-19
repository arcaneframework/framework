// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ZstdDataCompressor.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Service de compression utilisant la bibliothèque 'zstd'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IDataCompressor.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"
#include "arcane/IDeflateService.h"

#include <zstd.h>
// Nécessaire pour les versions de zstd antérieures à 1.5.6.
#include <zstd_errors.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de compression utilisant la bibliothèque 'zstd'.
 */
class ZstdDataCompressor
: public AbstractService
, public IDataCompressor
{
 public:

  explicit ZstdDataCompressor(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_name(sbi.serviceInfo()->localName())
  {
  }

 public:

  void build() override {}
  String name() const override { return m_name; }
  Int64 minCompressSize() const override { return 512; }
  void compress(Span<const std::byte> values, Array<std::byte>& compressed_values) override
  {
    size_t input_size = values.size();
    size_t dest_capacity = ZSTD_compressBound(input_size);
    compressed_values.resize(dest_capacity);

    void* dest = compressed_values.data();
    const void* source = values.data();
    int compression_level = 1;
    size_t r = ZSTD_compress(dest, dest_capacity, source, input_size, compression_level);
    if (ZSTD_isError(r)) {
      ZSTD_ErrorCode err_code = ZSTD_getErrorCode(r);
      ARCANE_THROW(IOException, "IO error during compression r={0} msg={1}", err_code, ZSTD_getErrorString(err_code));
    }
    size_t dest_len = r;
    if (input_size > 0) {
      Real ratio = (static_cast<double>(dest_len) * 100.0) / static_cast<double>(input_size);
      info(5) << "ZSTD compress r=" << r << " source_len=" << input_size
              << " dest_len=" << dest_len << " ratio=" << ratio;
    }
    compressed_values.resize(dest_len);
  }

  void decompress(Span<const std::byte> compressed_values, Span<std::byte> values) override
  {
    void* dest = values.data();
    size_t dest_len = values.size();

    const void* source = compressed_values.data();
    size_t source_len = compressed_values.size();

    size_t r = ZSTD_decompress(dest, dest_len, source, source_len);
    info(5) << "ZSTD decompress r=" << r << " source_len=" << source_len << " dest_len=" << dest_len;
    if (ZSTD_isError(r)) {
      ZSTD_ErrorCode err_code = ZSTD_getErrorCode(r);
      ARCANE_THROW(IOException, "IO error during decompression r={0} msg={1}", err_code, ZSTD_getErrorString(err_code));
    }
  }

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ZstdDataCompressor,
                        ServiceProperty("zstdDataCompressor", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IDataCompressor));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
