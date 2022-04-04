//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Xml;

namespace Arcane.VariableComparer
{
  public static class Utils
  {
    internal static IDeflater CreateDeflater(string service_name)
    {
      IDeflater deflater = null;
      if (service_name == "Bzip2" || service_name == "Bzip2DataCompressor") {
        deflater = new Bzip2Deflater();
      }
      else if (service_name == "LZ4" || service_name == "LZ4DataCompressor") {
        deflater = new LZ4Deflater();
      }
      else
        throw new ApplicationException("Can only handle 'Bzip2', 'LZ4', 'Bzip2DataCompressor' or 'LZ4DataCompressor' deflater-service");
      return deflater;
    }
    internal static IDeflater CreateOptionalDeflater(XmlElement doc_element)
    {
      string deflater_service = doc_element.GetAttribute("deflater-service");
      if (String.IsNullOrEmpty(deflater_service))
        return null;
      IDeflater deflater = CreateDeflater(deflater_service);
      
      // Cette valeur doit etre cohérente avec celle dans BasicReaderWriter.cc
      int deflate_min_size = 512;

      // Cette valeur n'est disponible qu'à partir de la version 3.0 de Arcane
      // et si elle existe, elle remplace DEFLATE_MIN_SIZE.
      string min_compress_size_str = doc_element.GetAttribute("min-compress-size");
      if (!String.IsNullOrEmpty(min_compress_size_str))
        deflate_min_size = int.Parse(min_compress_size_str);
      deflater.MinCompressSize = deflate_min_size;
      return deflater;
    }
  }
}