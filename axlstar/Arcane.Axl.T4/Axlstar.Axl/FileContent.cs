//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Text;

namespace Axlstar.Axl
{
  public class FileContent
  {
    public string Content { get; }
    public int Length { get; }
    public string Compression { get; }

    public FileContent()
    {
      Content = String.Empty;
      Compression = String.Empty;
    }

    public FileContent(byte[] bytes,string compression)
    {
      Content = Convert.ToBase64String (bytes);
      Length = Content.Length;
      Compression = compression;
    }
    /// <summary>
    /// Génère une chaîne de caractères multi-ligne compatible
    /// avec le C++ et le C#.
    /// </summary>
    /// <returns>The as multi string.</returns>
    public string ContentAsMultiString()
    {
      if (String.IsNullOrEmpty(Content))
        return "\"\"";
      StringBuilder sb = new StringBuilder();
      int length = Content.Length;
      const int block_size = 80;
      for (int i = 0; i < length; i += block_size) {
        int max_pos = Math.Min(i + block_size, length);
        sb.AppendFormat("\"{0}\"\n", Content.Substring(i, max_pos-i));
      }
      return sb.ToString();
    }
  }
}
